"""Download a subset of LIDC-IDRI CT series from TCIA.

Usage:
    pip install tcia_utils
    python3 download_lidc_subset.py --n_patients 50 --out_dir /home/lbw/project/LIDC-IDRI/DICOM

Steps:
    1. Query TCIA for all LIDC-IDRI series (CT modality only)
    2. Collect unique patient IDs, take the first N
    3. Download their CT series into out_dir/{SeriesInstanceUID}/*.dcm
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_patients", type=int, default=50,
                        help="Number of patients to download (default: 50 ≈ 6 GB)")
    parser.add_argument("--out_dir", default="/home/lbw/project/LIDC-IDRI/DICOM",
                        help="Output directory for raw DICOM files")
    parser.add_argument("--start", type=int, default=1,
                        help="Start patient number (default: 1)")
    args = parser.parse_args()

    try:
        from tcia_utils import nbia
    except ImportError:
        print("ERROR: tcia_utils not installed. Run: pip install tcia_utils")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    print("Querying TCIA for LIDC-IDRI CT series...")
    all_series = nbia.getSeries(collection="LIDC-IDRI")
    print(f"Total series (all modalities): {len(all_series)}")

    # Filter CT only
    ct_series = [s for s in all_series if s.get("Modality") == "CT"]
    print(f"CT series only: {len(ct_series)}")

    # Collect unique patient IDs in sorted order, take start..start+n_patients
    unique_pids = sorted({s["PatientID"] for s in ct_series})
    end_idx = args.start - 1 + args.n_patients
    target_pids = set(unique_pids[args.start - 1 : end_idx])
    print(f"Target patients ({len(target_pids)}): {sorted(target_pids)[0]} → {sorted(target_pids)[-1]}")

    # Filter series to target patients
    subset = [s for s in ct_series if s["PatientID"] in target_pids]
    print(f"CT series to download: {len(subset)}")

    total_bytes = sum(s.get("FileSize", 0) for s in subset)
    print(f"Estimated size: {total_bytes / 1e9:.1f} GB")
    print(f"\nDownloading to: {args.out_dir}\n")

    nbia.downloadSeries(subset, path=args.out_dir)

    print(f"\nDone. Downloaded {len(subset)} CT series for {len(target_pids)} patients.")
    print(f"Next: python3 ../classification_cnn/reorganize_dicom.py "
          f"--src {args.out_dir} --dst {args.out_dir}_organized")


if __name__ == "__main__":
    main()
