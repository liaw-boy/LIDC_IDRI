"""Reorganize the remaining 507 DICOM series that are missing from DICOM_organized.

Run: python3 reorganize_missing.py
"""
import os, shutil
from pathlib import Path
import pydicom
from concurrent.futures import ThreadPoolExecutor, as_completed

SRC = Path('/home/lbw/project/LIDC-IDRI/DICOM')
DST = Path('/home/lbw/project/LIDC-IDRI/DICOM_organized')
WORKERS = 8

def process_series(series_uid: str) -> tuple[str | None, int]:
    series_dir = SRC / series_uid
    if not series_dir.exists():
        return None, 0
    dcm_files = list(series_dir.glob('*.dcm'))
    if not dcm_files:
        return None, 0
    try:
        ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
        pid = f"LIDC-IDRI-{int(str(ds.PatientID).split('-')[-1]):04d}"
    except Exception:
        return None, 0
    target_dir = DST / pid / series_uid
    target_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for f in dcm_files:
        dst_file = target_dir / f.name
        if not dst_file.exists():
            shutil.copy2(str(f), str(dst_file))
        moved += 1
    return pid, moved

def main():
    import pylidc as pl
    org = set(os.listdir(DST))
    raw_series = set(os.listdir(SRC))
    scans = pl.query(pl.Scan).all()
    missing_series = [
        s.series_instance_uid
        for s in scans
        if s.patient_id not in org and s.series_instance_uid in raw_series
    ]
    print(f"Reorganizing {len(missing_series)} series into {DST}")
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(process_series, uid): uid for uid in missing_series}
        for fut in as_completed(futs):
            pid, n = fut.result()
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(missing_series)} done")
    total = len(list(DST.iterdir()))
    print(f"Done. Total patients in DICOM_organized: {total}")

if __name__ == '__main__':
    main()
