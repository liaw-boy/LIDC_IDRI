"""Train YOLOv11x on the nodule dataset (author's screenshot: YOLO("yolo11x.pt"), epochs=350, imgsz=640, batch=16, SGD,
workers=10, patience=20, pretrained=False). With the ultralytics of that time a .pt model + pretrained=False still started from
the COCO weights, so we start from yolo11x.pt explicitly (current ultralytics honours pretrained=False literally).
batch=8 because batch=16 does not fit an 11 GB GPU. Then evaluate val/test splits."""
import json, sys
from ultralytics import YOLO
D = '/mnt/legend800/lbw_project/lidc_data'
epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 350
m = YOLO('yolo11x.pt')
m.train(data=f'{D}/yolo_ds/data.yaml', epochs=epochs, imgsz=640, batch=8, device=0, optimizer='SGD',
        pretrained=True, amp=True, workers=8, project=f'{D}/yolo_run', name='x', exist_ok=True,
        plots=True, patience=20)
best = YOLO(f'{D}/yolo_run/x/weights/best.pt')
out = {}
for sp in ('val', 'test'):
    r = best.val(data=f'{D}/yolo_ds/data.yaml', split=sp, imgsz=640, batch=8, plots=False)
    out[sp] = dict(precision=float(r.box.mp), recall=float(r.box.mr), map50=float(r.box.map50), map50_95=float(r.box.map))
print(json.dumps(out, indent=1)); json.dump(out, open(f'{D}/yolo_run/result.json', 'w'), indent=1)
