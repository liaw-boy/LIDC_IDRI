"""Train the final Dual Input Nodule Classifier (report describes a simplified version of it). Report's Dual Input Nodule Classifier (GUI NoduleClassifier: ROI 32x32 + full CT 640x640,
2048+3200=5248 -> 256 -> 128 -> 2) on benign/malignant slices built by 03_build_classification.py.

Preprocessing: gray -> (x/255-0.5)/0.5 for BOTH paths (found by probing the shipped weights: AUC 0.99 with this vs 0.84 with plain /255;
the original GUI feeds plain /255, a train/inference mismatch); full CT resized to 640.
Split is random at slice level (the report itself notes slices of one patient can land in different splits).
"""
import argparse, glob, os, sys, json, time
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, confusion_matrix
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ref'))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PyQt5.QtWidgets import QApplication; _app = QApplication([])
from cnn_detector_v1_att import NoduleClassifier   # residual+CBAM attention variant = the real final model (report shows a simplified diagram)

p = argparse.ArgumentParser()
p.add_argument('--data', default='/mnt/legend800/lbw_project/lidc_data/cls_data')
p.add_argument('--out', default='/mnt/legend800/lbw_project/lidc_data/cls_run')
p.add_argument('--epochs', type=int, default=60)
p.add_argument('--bs', type=int, default=32)
p.add_argument('--lr', type=float, default=1e-4)
p.add_argument('--seed', type=int, default=42)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = 'cuda'

items = []
for lab, cls in enumerate(['benign', 'malignant']):
    for r in sorted(glob.glob(f'{a.data}/{cls}/*_roi.png')):
        items.append((r, r.replace('_roi.png', '_ct.png'), lab))
print('samples', len(items), 'malignant', sum(i[2] for i in items), flush=True)
idx = np.random.RandomState(a.seed).permutation(len(items))
n = len(idx); n_te = int(n * .1); n_va = int(n * .2)
split = {'test': idx[:n_te], 'val': idx[n_te:n_te + n_va], 'train': idx[n_te + n_va:]}
json.dump({k: [items[i][0] for i in v] for k, v in split.items()}, open(f'{a.out}/split.json', 'w'))

def load(ids):
    roi = np.stack([cv2.imread(items[i][0], 0) for i in ids])            # N,32,32
    ct = np.stack([cv2.imread(items[i][1], 0) for i in ids])             # N,512,512 uint8
    y = np.array([items[i][2] for i in ids])
    return torch.from_numpy(roi), torch.from_numpy(ct), torch.from_numpy(y)
data = {k: load(v) for k, v in split.items()}
print({k: len(v[2]) for k, v in data.items()}, flush=True)

def prep(roi, ct):
    r = roi.to(dev).float().div(255).sub(.5).div(.5).unsqueeze(1)
    c = ct.to(dev).float().div(255).sub(.5).div(.5).unsqueeze(1)
    c = F.interpolate(c, size=(640, 640), mode='bilinear', align_corners=False)
    return r, c

def augment(r, c):
    if np.random.rand() < .5: r, c = r.flip(3), c.flip(3)
    if np.random.rand() < .5: r, c = r.flip(2), c.flip(2)
    return r, c

@torch.no_grad()
def evaluate(split_name):
    m.eval(); roi, ct, y = data[split_name]; P = []
    for i in range(0, len(y), 64):
        r, c = prep(roi[i:i+64], ct[i:i+64])
        with torch.autocast('cuda', dtype=torch.float16):
            P.append(F.softmax(m(r, c)[0].float(), 1)[:, 1].cpu())
    P = torch.cat(P).numpy(); yy = y.numpy(); pred = (P > .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yy, pred).ravel()
    return dict(acc=(tp + tn) / len(yy), recall=tp / (tp + fn), spec=tn / (tn + fp),
                auc=roc_auc_score(yy, P), fp=int(fp), fn=int(fn), n=len(yy))

m = NoduleClassifier().to(dev)
opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
scaler = torch.cuda.amp.GradScaler()
roi, ct, y = data['train']; best = -1
for ep in range(a.epochs):
    m.train(); perm = torch.randperm(len(y)); tl = 0; t0 = time.time()
    for i in range(0, len(y) - a.bs + 1, a.bs):
        b = perm[i:i + a.bs]; r, c = augment(*prep(roi[b], ct[b]))
        with torch.autocast('cuda', dtype=torch.float16):
            loss = F.cross_entropy(m(r, c)[0].float(), y[b].to(dev))
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        tl += loss.item() * len(b)
    sched.step(); v = evaluate('val')
    print(f"ep{ep+1:3d} loss {tl/len(y):.4f} | val acc {v['acc']:.4f} auc {v['auc']:.4f} fp {v['fp']} fn {v['fn']} | {time.time()-t0:.0f}s", flush=True)
    if v['auc'] > best:
        best = v['auc']; torch.save(m.state_dict(), f'{a.out}/dual_input_best_auc_model.pth')
m.load_state_dict(torch.load(f'{a.out}/dual_input_best_auc_model.pth'))
res = {'val': evaluate('val'), 'test': evaluate('test')}
print(json.dumps(res, indent=1)); json.dump(res, open(f'{a.out}/result.json', 'w'), indent=1)
