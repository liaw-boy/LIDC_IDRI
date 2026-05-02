"""LIDC CSV dataset that stacks 3 adjacent slices into a 2.5D input.

Each __getitem__ returns:
  - roi_3ch: (3, roi_size, roi_size) — slice t-1, t, t+1 stacked as channels
  - ctx_3ch: (3, ctx_size, ctx_size) — same idea for context
  - label  : 0=benign, 1=malignant
  - aux    : (3,) auxiliary attribute targets in [0, 1] (lobulation, spiculation, margin)

If t-1 or t+1 doesn't exist (slice is at the edge), we duplicate slice t.
"""
from __future__ import annotations

import os
import random
import re
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


_SLICE_RE = re.compile(r"slice-(\d+)\.png$")


def _slice_index(path: str) -> int:
    m = _SLICE_RE.search(path)
    return int(m.group(1)) if m else -1


def _build_slice_lookup(df: pd.DataFrame) -> dict[tuple[str, int], dict[int, dict]]:
    """Return {(patient_id, nodule_id): {slice_idx: row_dict}} for fast neighbor lookup."""
    lookup: dict[tuple[str, int], dict[int, dict]] = defaultdict(dict)
    for r in df.to_dict("records"):
        key = (r["patient_id"], int(r["nodule_id"]))
        lookup[key][_slice_index(r["roi_path"])] = r
    return lookup


class LIDC25DDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        slice_lookup: dict[tuple[str, int], dict[int, dict]],
        roi_size: int = 64,
        ctx_size: int = 128,
        augment: bool = False,
    ):
        self.rows = rows
        self.lookup = slice_lookup
        self.roi_size = roi_size
        self.ctx_size = ctx_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def _load_gray(self, path: str, size: int) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return np.zeros((size, size), dtype=np.uint8)
        if img.shape != (size, size):
            img = cv2.resize(img, (size, size))
        return img

    def _stack_neighbors(self, row: dict, kind: str) -> np.ndarray:
        """Return (3, H, W) uint8 stack of [prev, this, next] for roi or ctx."""
        size = self.roi_size if kind == "roi" else self.ctx_size
        path_key = "roi_path" if kind == "roi" else "ctx_path"
        sib = self.lookup[(row["patient_id"], int(row["nodule_id"]))]
        idx = _slice_index(row[path_key])

        this_img = self._load_gray(row[path_key], size)
        prev_row = sib.get(idx - 1)
        next_row = sib.get(idx + 1)
        prev_img = self._load_gray(prev_row[path_key], size) if prev_row else this_img
        next_img = self._load_gray(next_row[path_key], size) if next_row else this_img
        return np.stack([prev_img, this_img, next_img], axis=0)

    def _augment_stack(self, stack: np.ndarray) -> np.ndarray:
        # Same spatial transform across 3 channels to preserve adjacency
        if random.random() > 0.5:
            stack = stack[:, :, ::-1].copy()
        if random.random() > 0.5:
            stack = stack[:, ::-1, :].copy()
        if random.random() > 0.3:
            angle = random.uniform(-15, 15)
            h, w = stack.shape[1:]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            for c in range(stack.shape[0]):
                stack[c] = cv2.warpAffine(stack[c], M, (w, h),
                                          borderMode=cv2.BORDER_REFLECT)
        if random.random() > 0.3:
            factor = random.uniform(0.85, 1.15)
            stack = np.clip(stack.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return stack

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        roi_stack = self._stack_neighbors(r, "roi")
        ctx_stack = self._stack_neighbors(r, "ctx")

        if self.augment:
            roi_stack = self._augment_stack(roi_stack)
            ctx_stack = self._augment_stack(ctx_stack)

        roi_t = (torch.from_numpy(roi_stack).float() / 255.0 - 0.5) / 0.5
        ctx_t = (torch.from_numpy(ctx_stack).float() / 255.0 - 0.5) / 0.5

        # Aux targets normalized to [0, 1] from LIDC 1-5 scale
        aux = torch.tensor([
            (float(r.get("lobulation", 1.0)) - 1.0) / 4.0,
            (float(r.get("spiculation", 1.0)) - 1.0) / 4.0,
            (float(r.get("margin", 1.0)) - 1.0) / 4.0,
        ], dtype=torch.float32).clamp(0.0, 1.0)

        return roi_t, ctx_t, int(r["label"]), aux


def create_25d_loaders(
    csv_path: str,
    batch_size: int = 16,
    roi_size: int = 64,
    ctx_size: int = 128,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    num_workers: int = 4,
    seed: int = 42,
):
    """Patient-level split (same as 2D dataloader) → no leakage between sets."""
    df = pd.read_csv(csv_path)
    pids = sorted(df["patient_id"].unique())
    random.seed(seed)
    random.shuffle(pids)
    n = len(pids)
    n_tr = int(n * train_ratio)
    n_va = int(n * val_ratio)
    tr_p = set(pids[:n_tr])
    va_p = set(pids[n_tr:n_tr + n_va])
    te_p = set(pids[n_tr + n_va:])
    print(f"Patients — Train:{len(tr_p)}, Val:{len(va_p)}, Test:{len(te_p)}")

    lookup = _build_slice_lookup(df)

    def _rows(ps: set[str]) -> list[dict]:
        return df[df["patient_id"].isin(ps)].to_dict("records")

    tr_rows, va_rows, te_rows = _rows(tr_p), _rows(va_p), _rows(te_p)
    print(f"Slices  — Train:{len(tr_rows)}, Val:{len(va_rows)}, Test:{len(te_rows)}")

    tr_ds = LIDC25DDataset(tr_rows, lookup, roi_size, ctx_size, augment=True)
    va_ds = LIDC25DDataset(va_rows, lookup, roi_size, ctx_size, augment=False)
    te_ds = LIDC25DDataset(te_rows, lookup, roi_size, ctx_size, augment=False)

    tr_l = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True)
    va_l = DataLoader(va_ds, batch_size=batch_size, shuffle=False,
                      num_workers=max(1, num_workers // 2))
    te_l = DataLoader(te_ds, batch_size=batch_size, shuffle=False,
                      num_workers=max(1, num_workers // 2))
    return tr_l, va_l, te_l
