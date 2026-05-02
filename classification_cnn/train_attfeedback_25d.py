"""2.5D Attribute Feedback training — adjacent 3-slice multi-view input.

Identical to train_attfeedback.py except:
  - Uses LIDC25DDataset (each sample = stacked [t-1, t, t+1] slices)
  - NoduleClassifier built with in_channels=3 (ROI + CTX both)
  - Output checkpoint: models/attfb25d_best_f1.pth
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from torch.amp import GradScaler, autocast

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gui_app.nodule_classifier import NoduleClassifier
from classification_cnn.lidc_csv_dataset_25d import create_25d_loaders
from classification_cnn.train_attfeedback import (AUX_WEIGHTS,
                                                   AttributeFeedbackHead,
                                                   weighted_aux_loss)

CSV_PATH = os.environ.get(
    "LIDC_LABELS_CSV",
    "/home/lbw/project/LIDC-IDRI/nodules_hires/labels_multitask.csv",
)
OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)


def train_one_epoch(model, loader, optimizer, scaler, device, aux_weight=0.3):
    model.train()
    total_loss = correct = n = 0
    ce = nn.CrossEntropyLoss()
    for roi, ctx, labels, aux_targets in loader:
        roi, ctx, labels, aux_targets = (roi.to(device), ctx.to(device),
                                         labels.to(device), aux_targets.to(device))
        optimizer.zero_grad()
        with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
            logits, aux_out = model(roi, ctx)
            loss_cls = ce(logits, labels)
            loss_aux = weighted_aux_loss(aux_out, aux_targets, AUX_WEIGHTS)
            loss = loss_cls + aux_weight * loss_aux
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
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
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    auc = roc_auc_score(all_labels, all_probs)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    return acc, auc, rec, prec, f1, cm


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("=== 2.5D AttFB training (3 adjacent slices stacked) ===")

    tr_loader, va_loader, te_loader = create_25d_loaders(
        CSV_PATH, batch_size=16, roi_size=64, ctx_size=128, num_workers=4)

    backbone = NoduleClassifier(in_channels=3)  # ← 2.5D entry layer
    model = AttributeFeedbackHead(backbone, n_aux=3).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-6)
    scaler = GradScaler()

    best_auc = best_f1 = 0
    patience = 0
    MAX_PATIENCE = 25
    EPOCHS = 80

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, tr_loader, optimizer, scaler, device)
        scheduler.step()
        acc, auc, rec, prec, f1, cm = evaluate(model, va_loader, device)
        flag = ""
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), f"{OUT_DIR}/attfb25d_best_auc.pth")
            flag += " [AUC]"
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), f"{OUT_DIR}/attfb25d_best_f1.pth")
            flag += " [F1]"
            patience = 0
        else:
            patience += 1
        print(f"Ep{epoch:3d} | loss={tr_loss:.4f} acc={tr_acc:.3f} | "
              f"val acc={acc:.3f} AUC={auc:.4f} Rec={rec:.3f} F1={f1:.3f}{flag}")
        if patience >= MAX_PATIENCE:
            print(f"Early stop at epoch {epoch}")
            break

    print("\n=== 最終測試結果 ===")
    for name, path in [("AUC最佳", f"{OUT_DIR}/attfb25d_best_auc.pth"),
                       ("F1最佳",  f"{OUT_DIR}/attfb25d_best_f1.pth")]:
        m2 = AttributeFeedbackHead(NoduleClassifier(in_channels=3), n_aux=3).to(device)
        m2.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        acc, auc, rec, prec, f1, cm = evaluate(m2, te_loader, device)
        print(f"{name}: Acc={acc:.4f} AUC={auc:.4f} Rec={rec:.4f} Prec={prec:.4f} F1={f1:.4f}")
        print(f"  CM: {cm.tolist()}")


if __name__ == "__main__":
    main()
