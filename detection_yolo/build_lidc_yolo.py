"""Build LIDC-native YOLO training dataset from existing CTX crops.

Each CTX crop is 256x256 with a known nodule near center.
Generate YOLO labels:
  class 0 (nodule), bbox center=(0.5±jitter, 0.5±jitter), w/h=(0.08~0.20)

Patient-level split (no leakage): 70/15/15 train/val/test.
"""
import os, csv, random, shutil
from collections import defaultdict
import pandas as pd

CSV_PATH = "/home/lbw/project/LIDC-IDRI/nodules_hires/labels_multitask.csv"
OUT_ROOT = "/home/lbw/project/LIDC-IDRI/yolo_lidc"
SEED = 42

random.seed(SEED)

# Load CSV
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows, {df.patient_id.nunique()} patients")

# Patient-level split (must match CNN split for fair end-to-end eval)
pids = sorted(df.patient_id.unique())
random.shuffle(pids)
n = len(pids); n_tr = int(n*0.70); n_va = int(n*0.15)
splits = {
    "train": set(pids[:n_tr]),
    "val":   set(pids[n_tr:n_tr+n_va]),
    "test":  set(pids[n_tr+n_va:]),
}
for k, v in splits.items():
    print(f"  {k}: {len(v)} patients")

# Prepare output dirs
for sp in ["train", "val", "test"]:
    os.makedirs(f"{OUT_ROOT}/images/{sp}", exist_ok=True)
    os.makedirs(f"{OUT_ROOT}/labels/{sp}", exist_ok=True)

# Generate labels — one per CTX image
counts = {sp: 0 for sp in splits}
for _, row in df.iterrows():
    src = row.ctx_path
    if not os.path.exists(src): continue
    pid = row.patient_id
    sp = next(k for k, v in splits.items() if pid in v)

    # Unique filename: patient_nodule_slice
    base = f"{pid}_n{row.nodule_id}_s{os.path.basename(src).replace('.png','')}"
    img_dst = f"{OUT_ROOT}/images/{sp}/{base}.png"
    lbl_dst = f"{OUT_ROOT}/labels/{sp}/{base}.txt"

    # Symlink (no disk waste)
    if not os.path.exists(img_dst):
        os.symlink(src, img_dst)

    # YOLO label: class cx cy w h  (normalized 0~1)
    # CTX is 256px = 16 × nodule_half_size (from build_lidc_dataset.py)
    # → nodule occupies 1/8 of CTX width = 0.125 normalized
    # Strict center, fixed size — match the actual extraction geometry
    cx = 0.5
    cy = 0.5
    w = 0.125
    h = 0.125
    with open(lbl_dst, "w") as f:
        f.write(f"0 {cx:.5f} {cy:.5f} {w:.5f} {h:.5f}\n")
    counts[sp] += 1

print("\nGenerated labels:")
for sp, c in counts.items():
    print(f"  {sp}: {c}")

# Write dataset YAML
yaml_path = f"{OUT_ROOT}/data.yaml"
with open(yaml_path, "w") as f:
    f.write(f"""path: {OUT_ROOT}
train: images/train
val: images/val
test: images/test

nc: 1
names: ['nodule']
""")
print(f"\nDataset YAML: {yaml_path}")
print(f"Done. To fine-tune:\n  yolo detect train data={yaml_path} model=models/best.pt epochs=30 imgsz=256 batch=32")
