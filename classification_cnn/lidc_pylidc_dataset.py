"""LIDC-IDRI Dataset using pylidc — reads original DICOM files.

Requires:
    - pylidc configured to point at DICOM_organized/
    - pip install pylidc pydicom

Usage:
    Set PYLIDC_DICOM_PATH env var, or configure ~/.pylidcrc:
        [dicom]
        path = /home/lbw/project/LIDC-IDRI/DICOM_organized

Returns (roi_tensor, full_ct_tensor, label):
    - roi_tensor:     (1, 32, 32)   nodule crop, HU normalized
    - full_ct_tensor: (1, 640, 640) full slice, HU normalized
    - label:          0=benign, 1=malignant (avg radiologist score >=3)
"""

import os
import random
import configparser
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pylidc as pl
from pylidc.utils import consensus


WC, WW = -600, 1500
LO, HI = WC - WW // 2, WC + WW // 2


def _configure_pylidc(dicom_root: str):
    cfg_path = os.path.expanduser("~/.pylidcrc")
    cfg = configparser.ConfigParser()
    cfg["dicom"] = {"path": dicom_root}
    with open(cfg_path, "w") as f:
        cfg.write(f)


def _hu_norm(arr: np.ndarray) -> np.ndarray:
    clipped = np.clip(arr.astype(np.float32), LO, HI)
    return ((clipped - LO) / WW * 255).astype(np.uint8)


def build_nodule_list(min_anns: int = 3, min_diam_mm: float = 3.0) -> list[dict]:
    """Return list of {scan, ann_cluster, label, patient_id} for all usable nodules."""
    scans = pl.query(pl.Scan).all()
    nodules = []
    for scan in scans:
        clusters = scan.cluster_annotations()
        for anns in clusters:
            if len(anns) < min_anns:
                continue
            diams = [a.diameter for a in anns if a.diameter > 0]
            if not diams or max(diams) < min_diam_mm:
                continue
            scores = [a.malignancy for a in anns]
            avg = sum(scores) / len(scores)
            label = 1 if avg >= 3 else 0
            nodules.append({
                "scan": scan,
                "anns": anns,
                "label": label,
                "patient_id": scan.patient_id,
                "mal_avg": avg,
            })
    return nodules


def build_sample_list(nodules: list[dict]) -> list[dict]:
    """Expand each nodule into per-slice samples."""
    samples = []
    for nod in nodules:
        anns = nod["anns"]
        scan = nod["scan"]
        try:
            cmask, cbbox, _ = consensus(anns, clevel=0.5)
        except Exception:
            continue

        vol = scan.to_volume()
        z_lo, z_hi = cbbox[2].start, cbbox[2].stop
        x_lo, x_hi = cbbox[0].start, cbbox[0].stop
        y_lo, y_hi = cbbox[1].start, cbbox[1].stop
        cx = (x_lo + x_hi) // 2
        cy = (y_lo + y_hi) // 2
        r = max(int(max((x_hi - x_lo), (y_hi - y_lo)) / 2 * 1.5), 16)

        for z_idx in range(z_lo, z_hi):
            if 0 <= z_idx < vol.shape[2]:
                samples.append({
                    "vol": vol,
                    "z_idx": z_idx,
                    "cx": cx, "cy": cy, "r": r,
                    "label": nod["label"],
                    "patient_id": nod["patient_id"],
                })
    return samples


class LIDCPylIdcDataset(Dataset):
    def __init__(
        self,
        dicom_root: str,
        roi_size: int = 32,
        full_ct_size: int = 640,
        augment: bool = False,
        patient_ids: list = None,
        min_anns: int = 3,
        min_diam_mm: float = 3.0,
    ):
        _configure_pylidc(dicom_root)
        self.roi_size = roi_size
        self.full_ct_size = full_ct_size
        self.augment = augment

        nodules = build_nodule_list(min_anns, min_diam_mm)
        if patient_ids is not None:
            nodules = [n for n in nodules if n["patient_id"] in patient_ids]

        self.samples = build_sample_list(nodules)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        vol, z_idx = s["vol"], s["z_idx"]
        cx, cy, r = s["cx"], s["cy"], s["r"]
        label = s["label"]

        slice_hu = vol[:, :, z_idx]
        img = _hu_norm(slice_hu)

        if self.augment:
            img = self._augment(img)

        x1 = max(0, cx - r);  x2 = min(img.shape[0], cx + r)
        y1 = max(0, cy - r);  y2 = min(img.shape[1], cy + r)
        crop = img[x1:x2, y1:y2]
        if crop.size == 0:
            crop = img

        roi  = cv2.resize(crop, (self.roi_size, self.roi_size))
        full = cv2.resize(img,  (self.full_ct_size, self.full_ct_size))

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


def create_pylidc_loaders(
    dicom_root: str,
    batch_size: int = 32,
    roi_size: int = 32,
    full_ct_size: int = 640,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    min_anns: int = 3,
    min_diam_mm: float = 3.0,
):
    _configure_pylidc(dicom_root)
    nodules = build_nodule_list(min_anns, min_diam_mm)

    # Patient-level split
    all_pids = sorted(set(n["patient_id"] for n in nodules))
    random.seed(42)
    random.shuffle(all_pids)
    n = len(all_pids)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    train_pids = set(all_pids[:n_train])
    val_pids   = set(all_pids[n_train:n_train + n_val])
    test_pids  = set(all_pids[n_train + n_val:])

    print(f"Patients — Train:{len(train_pids)}, Val:{len(val_pids)}, Test:{len(test_pids)}")

    def _make(pids, aug):
        return LIDCPylIdcDataset(
            dicom_root, roi_size, full_ct_size, aug,
            list(pids), min_anns, min_diam_mm,
        )

    train_ds = _make(train_pids, True)
    val_ds   = _make(val_pids,   False)
    test_ds  = _make(test_pids,  False)

    counts = [0, 0]
    for s in train_ds.samples:
        counts[s["label"]] += 1
    print(f"Train samples:{len(train_ds)}, Val:{len(val_ds)}, Test:{len(test_ds)}")
    print(f"Train 良性={counts[0]}, 惡性={counts[1]}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader
