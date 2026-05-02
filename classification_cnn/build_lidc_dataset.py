"""Build LIDC-IDRI nodule dataset from original DICOM files via pylidc.

Usage:
    python3 build_lidc_dataset.py --dicom_dir /path/to/LIDC-IDRI/DICOM_organized --out_dir /path/to/output

Requires:
    pip install pylidc pydicom opencv-python numpy

Output structure:
    out_dir/
      labels.csv          (roi_path, ctx_path, patient_id, nodule_id, malignancy_avg, label)
      LIDC-IDRI-XXXX/
        nodule-N/
          roi/slice-000.png   ← 64×64 tight crop around nodule
          ctx/slice-000.png   ← 256×256 surrounding lung context

Label convention (jaeho3690):
    malignancy avg <=2 → 0 (benign)
    malignancy avg >=4 → 1 (malignant)
    avg 2~4            → SKIPPED (ambiguous)
"""

import argparse
import configparser
import os
import csv
import numpy as np
import cv2
import pylidc as pl
from pylidc.utils import consensus
from collections import defaultdict


def configure_pylidc(dicom_dir: str):
    """Point pylidc at dicom_dir by writing ~/.pylidcrc."""
    cfg = configparser.ConfigParser()
    cfg["dicom"] = {"path": dicom_dir}
    cfg_path = os.path.expanduser("~/.pylidcrc")
    with open(cfg_path, "w") as f:
        cfg.write(f)


WC, WW = -600, 1500
LO, HI = WC - WW // 2, WC + WW // 2  # -1350 to 150 HU

ROI_SIZE = 64     # tight nodule crop
CTX_SIZE = 256    # surrounding lung context crop


def hu_to_uint8(vol: np.ndarray) -> np.ndarray:
    clipped = np.clip(vol.astype(np.float32), LO, HI)
    return ((clipped - LO) / WW * 255).astype(np.uint8)


def malignancy_label(anns) -> tuple[float, int | None]:
    """Return (avg_score, label). None means ambiguous — skip."""
    scores = [a.malignancy for a in anns]
    avg = sum(scores) / len(scores)
    if avg <= 2:
        return avg, 0
    if avg >= 4:
        return avg, 1
    return avg, None


def crop_and_resize(img: np.ndarray, cx: int, cy: int, half: int, size: int) -> np.ndarray:
    """Crop a square region centered at (cx, cy) with given half-side, resize to size×size."""
    h, w = img.shape
    x1 = max(0, cx - half);  x2 = min(w, cx + half)
    y1 = max(0, cy - half);  y2 = min(h, cy + half)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (size, size))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom_dir", default="/home/lbw/project/LIDC-IDRI/DICOM_CT_organized",
                        help="Root dir of LIDC-IDRI DICOM files (pylidc format)")
    parser.add_argument("--out_dir",   default="/home/lbw/project/LIDC-IDRI/nodules_hires",
                        help="Output directory for PNG crops + labels.csv")
    parser.add_argument("--min_anns",  type=int, default=3,
                        help="Min radiologist annotations required for a nodule")
    parser.add_argument("--min_diam",  type=float, default=3.0,
                        help="Min nodule diameter in mm")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Point pylidc at the correct DICOM dir for this batch
    configure_pylidc(args.dicom_dir)

    # Only process patients whose DICOM directory actually exists (supports subset downloads)
    available_pids = {
        d for d in os.listdir(args.dicom_dir)
        if os.path.isdir(os.path.join(args.dicom_dir, d)) and d.startswith("LIDC-IDRI-")
    }

    scans = pl.query(pl.Scan).all()
    scans = [s for s in scans if s.patient_id in available_pids]
    print(f"Found {len(available_pids)} patient dirs, processing {len(scans)} scans")

    rows = []
    skipped_small = 0
    skipped_few_anns = 0
    skipped_ambiguous = 0

    for scan in scans:
        pid = scan.patient_id
        nods = scan.cluster_annotations()

        for nod_idx, anns in enumerate(nods):
            if len(anns) < args.min_anns:
                skipped_few_anns += 1
                continue

            diams = [a.diameter for a in anns if a.diameter > 0]
            if not diams or max(diams) < args.min_diam:
                skipped_small += 1
                continue

            mal_avg, label = malignancy_label(anns)
            if label is None:
                skipped_ambiguous += 1
                continue

            try:
                cmask, cbbox, _ = consensus(anns, clevel=0.5)
            except Exception:
                continue

            vol = scan.to_volume()  # HU values, shape (H, W, Z)
            z_lo, z_hi = cbbox[2].start, cbbox[2].stop
            x_lo, x_hi = cbbox[0].start, cbbox[0].stop
            y_lo, y_hi = cbbox[1].start, cbbox[1].stop

            # Nodule center in (x, y) = (col, row)
            cx = (x_lo + x_hi) // 2
            cy = (y_lo + y_hi) // 2
            nodule_half = max((x_hi - x_lo), (y_hi - y_lo)) // 2
            nodule_half = max(nodule_half, 8)

            # ROI: 1.5× nodule bounding box → resize to 64×64
            roi_half = int(nodule_half * 1.5)

            # Context: 8× nodule half (large lung context) → resize to 256×256
            ctx_half = max(int(nodule_half * 8), 64)

            roi_dir = os.path.join(args.out_dir, pid, f"nodule-{nod_idx}", "roi")
            ctx_dir = os.path.join(args.out_dir, pid, f"nodule-{nod_idx}", "ctx")
            os.makedirs(roi_dir, exist_ok=True)
            os.makedirs(ctx_dir, exist_ok=True)

            saved = 0
            for z_idx in range(z_lo, z_hi):
                if z_idx < 0 or z_idx >= vol.shape[2]:
                    continue
                slice_hu = vol[:, :, z_idx]
                img8 = hu_to_uint8(slice_hu)  # shape (H, W)

                roi_img = crop_and_resize(img8, cx, cy, roi_half, ROI_SIZE)
                ctx_img = crop_and_resize(img8, cx, cy, ctx_half, CTX_SIZE)
                if roi_img is None or ctx_img is None:
                    continue

                roi_path = os.path.join(roi_dir, f"slice-{saved:03d}.png")
                ctx_path = os.path.join(ctx_dir, f"slice-{saved:03d}.png")
                cv2.imwrite(roi_path, roi_img)
                cv2.imwrite(ctx_path, ctx_img)

                rows.append({
                    "roi_path": roi_path,
                    "ctx_path": ctx_path,
                    "patient_id": pid,
                    "nodule_id": nod_idx,
                    "malignancy_avg": round(mal_avg, 3),
                    "label": label,
                })
                saved += 1

    csv_path = os.path.join(args.out_dir, "labels.csv")
    write_header = not os.path.exists(csv_path)  # append mode for batch pipeline
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "roi_path", "ctx_path", "patient_id", "nodule_id", "malignancy_avg", "label"
        ])
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    by_patient = defaultdict(set)
    for r in rows:
        by_patient[r["patient_id"]].add(r["label"])

    mal = sum(1 for r in rows if r["label"] == 1)
    print(f"\nDone.")
    print(f"  Total slices : {len(rows)}")
    print(f"  Patients     : {len(by_patient)}")
    print(f"  Malignant    : {mal}  ({mal/len(rows)*100:.1f}%)" if rows else "  No data")
    print(f"  Skipped (small nodule)       : {skipped_small}")
    print(f"  Skipped (<{args.min_anns} radiologists)  : {skipped_few_anns}")
    print(f"  Skipped (ambiguous score 2~4): {skipped_ambiguous}")
    print(f"  Labels saved : {csv_path}")


if __name__ == "__main__":
    main()
