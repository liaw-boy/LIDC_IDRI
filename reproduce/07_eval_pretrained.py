"""Evaluate the author's shipped weights on the rebuilt data, using the same test split as 04_train_classifier.py.
Caveat: the shipped weights were trained on the author's own split, so some of these slices were seen in training;
this only checks that the rebuilt data has the same distribution as the original (numbers near the report => same data)."""
import json, os, sys
import cv2, numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, confusion_matrix
here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(here, 'ref'), os.path.join(here, '..', 'gui_app')]
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PyQt5.QtWidgets import QApplication; _app = QApplication([])
import cnn_detector_v1 as plain, cnn_detector_v1_att as att

RUN = sys.argv[1] if len(sys.argv) > 1 else '/mnt/legend800/lbw_project/lidc_data/cls_run'
MODELS = {  # name -> (class, weights)
    'shipped best_auc (plain dual-input)': (plain.NoduleClassifier, os.path.join(here, '..', 'models', 'dual_input_best_auc_model.pth')),
    'shipped final (residual+CBAM)': (att.NoduleClassifier, os.path.join(here, '..', 'models', 'dual_input_final_model.pth')),
}
rois = json.load(open(f'{RUN}/split.json'))['test']
y = np.array([1 if '/malignant/' in r else 0 for r in rois])
R = torch.from_numpy(np.stack([cv2.imread(r, 0) for r in rois]))
C = torch.from_numpy(np.stack([cv2.imread(r.replace('_roi.png', '_ct.png'), 0) for r in rois]))
out = {}
for name, (cls, w) in MODELS.items():
    if not os.path.exists(w): print('skip (missing weights):', name); continue
    m = cls().cuda().eval(); m.load_state_dict(torch.load(w, map_location='cuda'), strict=True); P = []
    with torch.no_grad():
        for i in range(0, len(y), 64):
            r = R[i:i+64].cuda().float().div(255).sub(.5).div(.5).unsqueeze(1)
            c = F.interpolate(C[i:i+64].cuda().float().div(255).sub(.5).div(.5).unsqueeze(1), size=(640, 640), mode='bilinear', align_corners=False)
            P.append(F.softmax(m(r, c)[0], 1)[:, 1].cpu())
    P = torch.cat(P).numpy(); pred = (P > .5).astype(int); tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    out[name] = dict(n=len(y), acc=(tp+tn)/len(y), recall=tp/(tp+fn), spec=tn/(tn+fp), auc=roc_auc_score(y, P), fp=int(fp), fn=int(fn))
    print(name, json.dumps(out[name]))
json.dump(out, open(f'{RUN}/pretrained_eval.json', 'w'), indent=1)
