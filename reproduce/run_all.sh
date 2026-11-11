#!/bin/bash
# Waits for the TCIA download, then builds datasets and trains both models.
D=/mnt/legend800/lbw_project/lidc_data; R=/mnt/legend800/lbw_project/LIDC_IDRI/reproduce
cd $D
until grep -q '^DONE' download.log; do sleep 30; done
echo "[$(date)] download done"
rm -rf tree cls_data yolo_ds
python3 $R/01_build_tree.py
python3 $R/03_build_classification.py > build_cls.log 2>&1 && echo "[$(date)] cls data built"
python3 $R/02_build_detection.py > build_det.log 2>&1 && echo "[$(date)] det data built"
python3 $R/05_make_yolo_dataset.py
python3 $R/04_train_classifier.py --epochs 60 > train_cls.log 2>&1; echo "[$(date)] classifier done"
python3 $R/06_train_yolo.py 350 > train_yolo.log 2>&1; echo "[$(date)] yolo done"
