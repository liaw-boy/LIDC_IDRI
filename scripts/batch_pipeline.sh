#!/bin/bash
# 單批次處理：下載 → 整理 → 提取 → 刪除 DICOM
# 每次只處理一批，磁碟上只留當批 DICOM（處理完即刪）
#
# Usage:
#   bash batch_pipeline.sh <start> <end>
#   bash batch_pipeline.sh 151 250
#
# 建議從外部循環呼叫，例如：
#   for start in 151 251 351 451 551 651 751 851 951; do
#       bash batch_pipeline.sh $start $((start+99))
#   done

set -e

START=${1:?請提供起始病人編號，例如: bash batch_pipeline.sh 151 250}
END=${2:?請提供結束病人編號}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CNN_DIR="$PROJECT_DIR/classification_cnn"

DICOM_RAW="/home/lbw/project/LIDC-IDRI/DICOM_BATCH"
DICOM_ORG="/home/lbw/project/LIDC-IDRI/DICOM_BATCH_organized"
OUT_DIR="/home/lbw/project/LIDC-IDRI/nodules_hires"

N_PATIENTS=$((END - START + 1))

echo "========================================"
echo "批次: LIDC-IDRI-$(printf '%04d' $START) → LIDC-IDRI-$(printf '%04d' $END)  ($N_PATIENTS 名病人)"
echo "磁碟可用: $(df -h /home/lbw | tail -1 | awk '{print $4}')"
echo "========================================"

# 清除上一批殘留（如果有）
rm -rf "$DICOM_RAW" "$DICOM_ORG"

# Step 1: 下載
echo "[1/4] 下載 CT DICOM..."
python3 "$SCRIPT_DIR/download_lidc_subset.py" \
    --start "$START" \
    --n_patients "$N_PATIENTS" \
    --out_dir "$DICOM_RAW"

echo "    下載完成，磁碟剩餘: $(df -h /home/lbw | tail -1 | awk '{print $4}')"

# Step 2: 整理目錄結構
echo "[2/4] 整理 DICOM 目錄..."
python3 "$CNN_DIR/reorganize_dicom.py" \
    --src "$DICOM_RAW" \
    --dst "$DICOM_ORG" \
    --workers 8

# Step 3: 提取結節 PNG（append 到同一個 labels.csv）
echo "[3/4] 提取結節 PNG..."
python3 "$CNN_DIR/build_lidc_dataset.py" \
    --dicom_dir "$DICOM_ORG" \
    --out_dir "$OUT_DIR"

# Step 4: 刪除本批 DICOM
echo "[4/4] 刪除 DICOM 釋放空間..."
rm -rf "$DICOM_RAW" "$DICOM_ORG"

echo ""
echo "批次完成！磁碟剩餘: $(df -h /home/lbw | tail -1 | awk '{print $4}')"
echo "累計切片數: $(($(wc -l < "$OUT_DIR/labels.csv") - 1)) 筆"
echo "========================================"
