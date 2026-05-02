"""LIDC-IDRI CSV Dataset Loader — reads labels.csv produced by build_lidc_dataset.py.

Each row in labels.csv: roi_path, ctx_path, patient_id, nodule_id, malignancy_avg, label

Returns (roi_tensor, ctx_tensor, label):
  - roi_tensor: (1, roi_size, roi_size)   tight 64×64 nodule crop
  - ctx_tensor: (1, ctx_size, ctx_size)   256×256 surrounding lung context
  - label:      0=benign, 1=malignant
"""

import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class LIDCCsvDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        roi_size: int = 64,
        ctx_size: int = 128,
        augment: bool = False,
    ):
        self.rows = rows
        self.roi_size = roi_size
        self.ctx_size = ctx_size
        self.augment = augment
        self.data_list = [(r["roi_path"], r["label"]) for r in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]

        roi = cv2.imread(r["roi_path"], cv2.IMREAD_GRAYSCALE)
        ctx = cv2.imread(r["ctx_path"], cv2.IMREAD_GRAYSCALE)

        if roi is None:
            roi = np.zeros((self.roi_size, self.roi_size), dtype=np.uint8)
        if ctx is None:
            ctx = np.zeros((self.ctx_size, self.ctx_size), dtype=np.uint8)

        if self.augment:
            roi, ctx = self._augment(roi, ctx)

        roi = cv2.resize(roi, (self.roi_size, self.roi_size))
        ctx = cv2.resize(ctx, (self.ctx_size, self.ctx_size))

        roi_t = (torch.from_numpy(roi).float().unsqueeze(0) / 255.0 - 0.5) / 0.5
        ctx_t = (torch.from_numpy(ctx).float().unsqueeze(0) / 255.0 - 0.5) / 0.5
        return roi_t, ctx_t, r["label"]

    def _augment(self, roi: np.ndarray, ctx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Apply same spatial transform to both crops
        if random.random() > 0.5:
            roi = cv2.flip(roi, 1)
            ctx = cv2.flip(ctx, 1)
        if random.random() > 0.5:
            roi = cv2.flip(roi, 0)
            ctx = cv2.flip(ctx, 0)
        if random.random() > 0.3:
            angle = random.uniform(-15, 15)
            for img in [roi, ctx]:
                h, w = img.shape
                M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            roi = cv2.warpAffine(roi, M, (roi.shape[1], roi.shape[0]), borderMode=cv2.BORDER_REFLECT)
            ctx = cv2.warpAffine(ctx, M, (ctx.shape[1], ctx.shape[0]), borderMode=cv2.BORDER_REFLECT)
        # Brightness jitter (applied independently)
        if random.random() > 0.3:
            factor = random.uniform(0.85, 1.15)
            roi = np.clip(roi.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        if random.random() > 0.3:
            factor = random.uniform(0.9, 1.1)
            ctx = np.clip(ctx.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return roi, ctx


def create_csv_loaders(
    csv_path: str,
    batch_size: int = 16,
    roi_size: int = 64,
    ctx_size: int = 128,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    num_workers: int = 4,
):
    """Patient-level split → no data leakage between train/val/test."""
    df = pd.read_csv(csv_path)

    all_pids = sorted(df["patient_id"].unique())
    random.seed(42)
    random.shuffle(all_pids)
    n = len(all_pids)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    train_pids = set(all_pids[:n_train])
    val_pids   = set(all_pids[n_train:n_train + n_val])
    test_pids  = set(all_pids[n_train + n_val:])

    print(f"Patients — Train:{len(train_pids)}, Val:{len(val_pids)}, Test:{len(test_pids)}")

    def _rows(pids):
        sub = df[df["patient_id"].isin(pids)]
        return sub[["roi_path", "ctx_path", "patient_id", "label"]].to_dict("records")

    train_rows = _rows(train_pids)
    val_rows   = _rows(val_pids)
    test_rows  = _rows(test_pids)

    train_ds = LIDCCsvDataset(train_rows, roi_size, ctx_size, augment=True)
    val_ds   = LIDCCsvDataset(val_rows,   roi_size, ctx_size, augment=False)
    test_ds  = LIDCCsvDataset(test_rows,  roi_size, ctx_size, augment=False)

    print(f"Samples — Train:{len(train_ds)}, Val:{len(val_ds)}, Test:{len(test_ds)}")
    counts = [0, 0]
    for _, lbl in train_ds.data_list:
        counts[lbl] += 1
    print(f"Train 良性={counts[0]}, 惡性={counts[1]}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=max(1, num_workers // 2))
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=max(1, num_workers // 2))

    return train_loader, val_loader, test_loader
