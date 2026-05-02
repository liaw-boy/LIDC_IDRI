"""V3: LIDC YOLO 数据集 + hard negatives.
正样本: CTX 256x256 + 中心 bbox (0.5, 0.5, 0.125, 0.125)
负样本: CTX 左上角 128x128 -> resize 256x256 + 空标签 (background)
比例: positives:negatives = 1:0.5 (避免 negatives 主导)
"""
import os, csv, random, shutil
from collections import defaultdict
import pandas as pd
import cv2

CSV_PATH = "/home/lbw/project/LIDC-IDRI/nodules_hires/labels_multitask.csv"
OUT_ROOT = "/home/lbw/project/LIDC-IDRI/yolo_lidc_v3"
SEED = 42

random.seed(SEED)

df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows, {df.patient_id.nunique()} patients")

pids = sorted(df.patient_id.unique())
random.shuffle(pids)
n = len(pids); n_tr = int(n*0.70); n_va = int(n*0.15)
splits = {
    "train": set(pids[:n_tr]),
    "val":   set(pids[n_tr:n_tr+n_va]),
    "test":  set(pids[n_tr+n_va:]),
}

# Clean
shutil.rmtree(OUT_ROOT, ignore_errors=True)
for sp in splits:
    os.makedirs(f"{OUT_ROOT}/images/{sp}", exist_ok=True)
    os.makedirs(f"{OUT_ROOT}/labels/{sp}", exist_ok=True)
NEG_DIR = f"{OUT_ROOT}/negatives_cache"
os.makedirs(NEG_DIR, exist_ok=True)

pos_counts = {sp: 0 for sp in splits}
neg_counts = {sp: 0 for sp in splits}

for _, row in df.iterrows():
    src = row.ctx_path
    if not os.path.exists(src): continue
    pid = row.patient_id
    sp = next(k for k, v in splits.items() if pid in v)

    # === Positive ===
    base = f"{pid}_n{row.nodule_id}_s{os.path.basename(src).replace('.png','')}"
    img_dst = f"{OUT_ROOT}/images/{sp}/{base}.png"
    lbl_dst = f"{OUT_ROOT}/labels/{sp}/{base}.txt"
    if not os.path.exists(img_dst):
        os.symlink(src, img_dst)
    with open(lbl_dst, "w") as f:
        f.write("0 0.5 0.5 0.125 0.125\n")
    pos_counts[sp] += 1

    # === Negative === (50% chance to add a negative for this sample)
    if random.random() < 0.5:
        ctx = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
        if ctx is None: continue
        # Take a corner (TL, TR, BL, BR) randomly, crop 128x128, upscale to 256x256
        corners = [(0, 0), (128, 0), (0, 128), (128, 128)]
        x0, y0 = random.choice(corners)
        crop = ctx[y0:y0+128, x0:x0+128]
        if crop.shape != (128, 128): continue
        neg_resized = cv2.resize(crop, (256, 256))
        neg_path = f"{NEG_DIR}/{base}_neg.png"
        cv2.imwrite(neg_path, neg_resized)

        # Symlink + EMPTY label = background image (YOLO ignores)
        neg_img_dst = f"{OUT_ROOT}/images/{sp}/{base}_neg.png"
        neg_lbl_dst = f"{OUT_ROOT}/labels/{sp}/{base}_neg.txt"
        if not os.path.exists(neg_img_dst):
            os.symlink(neg_path, neg_img_dst)
        # Write empty label file
        open(neg_lbl_dst, "w").close()
        neg_counts[sp] += 1

print("\nGenerated:")
for sp in splits:
    print(f"  {sp}: {pos_counts[sp]} pos + {neg_counts[sp]} neg = {pos_counts[sp]+neg_counts[sp]}")

with open(f"{OUT_ROOT}/data.yaml", "w") as f:
    f.write(f"""path: {OUT_ROOT}
train: images/train
val: images/val
test: images/test

nc: 1
names: ['nodule']
""")
print(f"\nDataset YAML: {OUT_ROOT}/data.yaml")
