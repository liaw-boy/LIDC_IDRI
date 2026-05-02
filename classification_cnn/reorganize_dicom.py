"""Reorganize TCIA-downloaded DICOM files into pylidc-compatible structure.

TCIA downloads as: DICOM/{SeriesInstanceUID}/{filename}.dcm
pylidc expects:    DICOM/LIDC-IDRI-XXXX/{SeriesInstanceUID}/{filename}.dcm

Usage:
    python3 reorganize_dicom.py \
        --src /home/lbw/project/LIDC-IDRI/DICOM \
        --dst /home/lbw/project/LIDC-IDRI/DICOM_organized
"""

import argparse
import os
import shutil
from pathlib import Path
import pydicom
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_patient_id(dcm_path: str) -> str | None:
    try:
        ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
        pid = str(ds.PatientID).strip()
        # Normalize to 4-digit format
        parts = pid.split("-")
        return f"LIDC-IDRI-{int(parts[-1]):04d}"
    except Exception:
        return None


def process_series(series_dir: Path, dst_root: Path) -> tuple[str, int]:
    dcm_files = list(series_dir.glob("*.dcm"))
    if not dcm_files:
        return str(series_dir), 0

    pid = get_patient_id(str(dcm_files[0]))
    if pid is None:
        return str(series_dir), 0

    target_dir = dst_root / pid / series_dir.name
    target_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    for f in dcm_files:
        dst_file = target_dir / f.name
        if not dst_file.exists():
            shutil.copy2(str(f), str(dst_file))
        moved += 1

    return pid, moved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="/home/lbw/project/LIDC-IDRI/DICOM",
                        help="TCIA download dir (SeriesUID subdirs)")
    parser.add_argument("--dst", default="/home/lbw/project/LIDC-IDRI/DICOM_organized",
                        help="Output dir (PatientID subdirs, pylidc compatible)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)
    dst_root.mkdir(parents=True, exist_ok=True)

    series_dirs = [d for d in src_root.iterdir() if d.is_dir()]
    print(f"Found {len(series_dirs)} series directories")

    patients = set()
    total_files = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_series, d, dst_root): d for d in series_dirs}
        done = 0
        for fut in as_completed(futs):
            pid, n = fut.result()
            done += 1
            if n > 0:
                patients.add(pid)
                total_files += n
            if done % 50 == 0:
                print(f"  {done}/{len(series_dirs)} series done, "
                      f"{len(patients)} patients, {total_files} files")

    print(f"\nDone: {len(patients)} patients, {total_files} DICOM files")
    print(f"Output: {args.dst}")


if __name__ == "__main__":
    main()
