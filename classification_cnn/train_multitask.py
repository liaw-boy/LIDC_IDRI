"""Multi-task training: malignancy classification + spiculation/margin/texture regression.
Auxiliary tasks force the backbone to learn clinically meaningful features → higher recall.
Output model is compatible with gui_app/model_manager.py (same NoduleClassifier weights).
"""
import os, sys, random, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, confusion_matrix
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gui_app.nodule_classifier import NoduleClassifier

CSV_PATH = os.environ.get(
    "LIDC_LABELS_CSV",
    "/home/lbw/project/LIDC-IDRI/nodules_hires/labels_multitask.csv"
)
OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Dataset ────────────────────────────────────────────────────────────────────
AUX_COLS = ["lobulation", "spiculation", "margin"]  # top-3 correlates with malignancy

class MultiTaskDataset(Dataset):
    def __init__(self, rows, roi_size=64, ctx_size=128, augment=False):
        self.rows = rows
        self.roi_size = roi_size
        self.ctx_size = ctx_size
        self.augment = augment

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        roi = cv2.imread(row["roi_path"], cv2.IMREAD_GRAYSCALE)
        ctx = cv2.imread(row["ctx_path"], cv2.IMREAD_GRAYSCALE)
        roi = cv2.resize(roi, (self.roi_size, self.roi_size)).astype(np.float32) / 255.0
        ctx = cv2.resize(ctx, (self.ctx_size, self.ctx_size)).astype(np.float32) / 255.0

        if self.augment:
            if random.random() > 0.5:
                roi = np.fliplr(roi).copy()
                ctx = np.fliplr(ctx).copy()
            if random.random() > 0.5:
                roi = np.flipud(roi).copy()
                ctx = np.flipud(ctx).copy()
            k = random.randint(0, 3)
            roi = np.rot90(roi, k).copy()
            ctx = np.rot90(ctx, k).copy()
            alpha = random.uniform(0.85, 1.15)
            roi = np.clip(roi * alpha, 0, 1)
            ctx = np.clip(ctx * alpha, 0, 1)

        roi_t = torch.tensor(roi).unsqueeze(0)
        ctx_t = torch.tensor(ctx).unsqueeze(0)
        label = torch.tensor(row["label"], dtype=torch.long)
        # Normalise aux targets to [0,1]
        aux = torch.tensor(
            [(row[c] - 1.0) / 4.0 for c in AUX_COLS], dtype=torch.float32
        )
        return roi_t, ctx_t, label, aux


def patient_split(csv_path, train_r=0.70, val_r=0.15, seed=42):
    df = pd.read_csv(csv_path)
    pids = sorted(df["patient_id"].unique())
    random.seed(seed); random.shuffle(pids)
    n = len(pids)
    n_tr = int(n * train_r); n_va = int(n * val_r)
    tr_p = set(pids[:n_tr]); va_p = set(pids[n_tr:n_tr+n_va]); te_p = set(pids[n_tr+n_va:])
    def rows(ps): return df[df.patient_id.isin(ps)].to_dict("records")
    print(f"Patients  Train:{len(tr_p)} Val:{len(va_p)} Test:{len(te_p)}")
    tr, va, te = rows(tr_p), rows(va_p), rows(te_p)
    print(f"Samples   Train:{len(tr)} Val:{len(va)} Test:{len(te)}")
    return tr, va, te


# ── Multi-task head ────────────────────────────────────────────────────────────
class MultiTaskHead(nn.Module):
    """Wraps NoduleClassifier and adds aux regression heads."""
    def __init__(self, backbone: NoduleClassifier, n_aux: int = 3):
        super().__init__()
        self.backbone = backbone
        feat_dim = 256  # after fusion_fc1
        self.aux_head = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_aux),
            nn.Sigmoid(),   # output in [0,1]
        )

    def forward(self, roi, ctx):
        # Run backbone up to fusion_fc1 manually
        # ROI branch
        x = self.backbone.roi_pool1(F.relu(self.backbone.roi_bn1(self.backbone.roi_conv1(roi))))
        x, _ = self.backbone.roi_att1(x)
        x = self.backbone.roi_pool2(x)
        x, _ = self.backbone.roi_att2(x)
        x = self.backbone.roi_pool3(x)
        x = self.backbone.roi_adaptive_pool(x)
        roi_feat = x.view(x.size(0), -1)

        # CTX branch
        y = self.backbone.full_ct_pool1(F.relu(self.backbone.full_ct_bn1(self.backbone.full_ct_conv1(ctx))))
        y, _ = self.backbone.full_ct_att1(y)
        y = self.backbone.full_ct_pool2(y)
        y, _ = self.backbone.full_ct_att2(y)
        y = self.backbone.full_ct_pool3(y)
        y, _ = self.backbone.full_ct_att3(y)
        y = self.backbone.full_ct_adaptive_pool(y)
        ctx_feat = y.view(y.size(0), -1)

        combined = torch.cat([roi_feat, ctx_feat], dim=1)
        shared = F.relu(self.backbone.fusion_bn1(self.backbone.fusion_fc1(combined)))

        # Main classification head
        dropped = self.backbone.fusion_dropout(shared)
        cls_feat = F.relu(self.backbone.fusion_bn2(self.backbone.fusion_fc2(dropped)))
        logits = self.backbone.fusion_fc3(cls_feat)

        # Aux regression head
        aux_out = self.aux_head(shared)
        return logits, aux_out


# ── Training ───────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scaler, device, aux_weight=0.3):
    model.train()
    total_loss = 0; correct = 0; n = 0
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    for roi, ctx, labels, aux_targets in loader:
        roi, ctx, labels, aux_targets = roi.to(device), ctx.to(device), labels.to(device), aux_targets.to(device)
        optimizer.zero_grad()
        with autocast(device_type='cuda' if device.type=='cuda' else 'cpu'):
            logits, aux_out = model(roi, ctx)
            loss_cls = ce(logits, labels)
            loss_aux = mse(aux_out, aux_targets)
            loss = loss_cls + aux_weight * loss_aux
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        n += len(labels)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    for roi, ctx, labels, _ in loader:
        roi, ctx, labels = roi.to(device), ctx.to(device), labels.to(device)
        logits, _ = model(roi, ctx)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = (probs > 0.4).long()
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
    acc  = sum(p==l for p,l in zip(all_preds, all_labels)) / len(all_labels)
    auc  = roc_auc_score(all_labels, all_probs)
    rec  = recall_score(all_labels, all_preds, zero_division=0)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    f1   = f1_score(all_labels, all_preds, zero_division=0)
    cm   = confusion_matrix(all_labels, all_preds)
    return acc, auc, rec, prec, f1, cm


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tr_rows, va_rows, te_rows = patient_split(CSV_PATH)

    tr_ds = MultiTaskDataset(tr_rows, augment=True)
    va_ds = MultiTaskDataset(va_rows, augment=False)
    te_ds = MultiTaskDataset(te_rows, augment=False)

    tr_loader = DataLoader(tr_ds, batch_size=16, shuffle=True,  num_workers=4, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    te_loader = DataLoader(te_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    backbone = NoduleClassifier()
    model = MultiTaskHead(backbone, n_aux=len(AUX_COLS)).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    scaler = GradScaler()

    best_auc = 0; best_f1 = 0; patience = 0; MAX_PATIENCE = 30
    EPOCHS = 150

    print(f"\n開始 Multi-task 訓練 ({EPOCHS} epochs)...")
    for epoch in range(1, EPOCHS+1):
        tr_loss, tr_acc = train_one_epoch(model, tr_loader, optimizer, scaler, device)
        scheduler.step()
        acc, auc, rec, prec, f1, cm = evaluate(model, va_loader, device)

        flag = ""
        if auc > best_auc:
            best_auc = auc
            torch.save(model.backbone.state_dict(), f"{OUT_DIR}/multitask_best_auc.pth")
            flag += " [AUC]"
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.backbone.state_dict(), f"{OUT_DIR}/multitask_best_f1.pth")
            flag += " [F1]"
            patience = 0
        else:
            patience += 1

        print(f"Ep{epoch:3d} | loss={tr_loss:.4f} acc={tr_acc:.3f} | "
              f"val acc={acc:.3f} AUC={auc:.4f} Rec={rec:.3f} F1={f1:.3f}{flag}")
        print(f"       CM: {cm.tolist()}")

        if patience >= MAX_PATIENCE:
            print(f"Early stop at epoch {epoch}")
            break

    # Final test evaluation
    print("\n=== 最終測試結果 ===")
    for name, path in [("AUC最佳", f"{OUT_DIR}/multitask_best_auc.pth"),
                        ("F1最佳",  f"{OUT_DIR}/multitask_best_f1.pth")]:
        backbone2 = NoduleClassifier()
        backbone2.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        m2 = MultiTaskHead(backbone2).to(device)
        acc, auc, rec, prec, f1, cm = evaluate(m2, te_loader, device)
        print(f"{name}: Acc={acc:.4f} AUC={auc:.4f} Rec={rec:.4f} Prec={prec:.4f} F1={f1:.4f}")
        print(f"  CM: {cm.tolist()}")

    # Deploy best model
    import shutil
    shutil.copy(f"{OUT_DIR}/multitask_best_f1.pth",
                "/home/lbw/project/Lung_Nodule_System/models/dual_input_final_model.pth")
    print("\n最佳模型已部署至 models/dual_input_final_model.pth")

if __name__ == "__main__":
    main()
