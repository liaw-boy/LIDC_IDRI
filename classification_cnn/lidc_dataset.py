"""LIDC-IDRI PNG Dataset Loader

Reads from the Kaggle `zhangweiled/lidcidri` structure:
  LIDC-IDRI-slices/
    LIDC-IDRI-XXXX/
      nodule-N/
        images/slice-M.png   <- 128×128 grayscale nodule crop

Label sources (in priority order):
  1. tcia-diagnosis-data-2012-04-20.xls  — biopsy/surgical ground truth (157 pts)
  2. LIDC XML annotations matched by filename (300 pts, majority vote ≥3 = malignant)

Returns (roi_tensor, full_ct_tensor, label):
  - roi_tensor:     (1, roi_size, roi_size)
  - full_ct_tensor: (1, full_ct_size, full_ct_size)
  - label:          0=benign, 1=malignant
"""

import os
import glob
import random
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from lxml import etree


def build_label_map(xml_root: str, diagnosis_xls: str = None) -> dict:
    """Return {patient_id_4digit: 0/1}.

    xml_root     : path to tcia-lidc-xml/ (contains numeric batch subdirs)
    diagnosis_xls: optional path to tcia-diagnosis-data-2012-04-20.xls
    """
    ns = {"ns": "http://www.nih.gov"}

    # --- Source 1: XML radiologist votes, matched by filename ---
    # XML files live at xml_root/{batch_num}/{patient_num:03d}.xml
    patient_votes: dict[str, list] = defaultdict(list)
    for xf in glob.glob(os.path.join(xml_root, "**", "*.xml"), recursive=True):
        basename = os.path.basename(xf).replace(".xml", "")
        try:
            n = int(basename)
            pid = f"LIDC-IDRI-{n:04d}"
        except ValueError:
            continue
        try:
            tree = etree.parse(xf)
        except Exception:
            continue
        mals = [
            int(m.text)
            for m in tree.findall(".//ns:malignancy", ns)
            if m.text and m.text.strip().isdigit()
        ]
        if mals:
            avg = sum(mals) / len(mals)
            patient_votes[pid].append(1 if avg >= 3 else 0)

    label_map = {
        pid: (1 if sum(votes) / len(votes) >= 0.5 else 0)
        for pid, votes in patient_votes.items()
    }

    # --- Source 2: Clinical diagnosis XLS (overrides radiologist votes) ---
    if diagnosis_xls and os.path.isfile(diagnosis_xls):
        try:
            import pandas as pd
            df = pd.read_excel(diagnosis_xls, engine="xlrd")
            pid_col = "TCIA Patient ID"
            diag_col = next(
                c for c in df.columns if "Patient Level" in c and "Diagnosis at" in c
            )
            for _, row in df.iterrows():
                pid = str(row[pid_col]).strip()
                diag = row[diag_col]
                if pd.notna(diag) and int(diag) != 0:
                    label_map[pid] = 1 if int(diag) >= 2 else 0
        except Exception as e:
            print(f"[WARN] Could not load diagnosis XLS: {e}")

    return label_map


class LIDCNoduleDataset(Dataset):
    """One sample = center slice of one nodule (ROI crop + upscaled full-view)."""

    def __init__(
        self,
        slices_root: str,
        xml_root: str,
        diagnosis_xls: str = None,
        roi_size: int = 32,
        full_ct_size: int = 640,
        augment: bool = False,
        patient_ids: list = None,
    ):
        self.roi_size = roi_size
        self.full_ct_size = full_ct_size
        self.augment = augment

        label_map = build_label_map(xml_root, diagnosis_xls)

        self.data_list = []
        all_pids = sorted(os.listdir(slices_root))
        if patient_ids is not None:
            all_pids = [p for p in all_pids if p in patient_ids]

        for pid in all_pids:
            if pid not in label_map:
                continue
            label = label_map[pid]
            nodule_dirs = glob.glob(os.path.join(slices_root, pid, "nodule-*"))
            for nd in sorted(nodule_dirs):
                slices = sorted(glob.glob(os.path.join(nd, "images", "slice-*.png")))
                if not slices:
                    continue
                center = slices[len(slices) // 2]
                self.data_list.append((center, label))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        path, label = self.data_list[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((128, 128), dtype=np.uint8)

        if self.augment:
            img = self._augment(img)

        roi  = cv2.resize(img, (self.roi_size, self.roi_size))
        full = cv2.resize(img, (self.full_ct_size, self.full_ct_size))

        roi_t  = (torch.from_numpy(roi).float().unsqueeze(0)  / 255.0 - 0.5) / 0.5
        full_t = (torch.from_numpy(full).float().unsqueeze(0) / 255.0 - 0.5) / 0.5

        return roi_t, full_t, label

    def _augment(self, img: np.ndarray) -> np.ndarray:
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
        if random.random() > 0.5:
            img = cv2.flip(img, 0)
        if random.random() > 0.3:
            angle = random.uniform(-15, 15)
            h, w = img.shape
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        if random.random() > 0.3:
            factor = random.uniform(0.85, 1.15)
            img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return img


def create_lidc_loaders(
    slices_root: str,
    xml_root: str,
    diagnosis_xls: str = None,
    # kept for backward compatibility, ignored
    sop_csv: str = None,
    batch_size: int = 32,
    roi_size: int = 32,
    full_ct_size: int = 640,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
):
    """Patient-level train/val/test split to avoid data leakage."""
    from torch.utils.data import DataLoader

    label_map_all = build_label_map(xml_root, diagnosis_xls)
    png_pids = set(os.listdir(slices_root))
    usable = [pid for pid in label_map_all if pid in png_pids]
    random.seed(42)
    random.shuffle(usable)

    n = len(usable)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    train_pids = usable[:n_train]
    val_pids   = usable[n_train:n_train + n_val]
    test_pids  = usable[n_train + n_val:]

    print(f"Train patients={len(train_pids)}, Val={len(val_pids)}, Test={len(test_pids)}")

    def _make(pids, aug):
        return LIDCNoduleDataset(
            slices_root, xml_root, diagnosis_xls,
            roi_size, full_ct_size, augment=aug, patient_ids=pids,
        )

    train_ds = _make(train_pids, True)
    val_ds   = _make(val_pids,   False)
    test_ds  = _make(test_pids,  False)

    print(f"Train samples={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    counts = [0, 0]
    for _, lbl in train_ds.data_list:
        counts[lbl] += 1
    print(f"Train 良性={counts[0]}, 惡性={counts[1]}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader
