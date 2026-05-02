"""Extract LIDC-IDRI nodule features and add to labels.csv for multi-task training."""
import pylidc as pl
import pandas as pd
import numpy as np
import os, sys

CSV_IN  = '/home/lbw/project/LIDC-IDRI/nodules_hires/labels.csv'
CSV_OUT = '/home/lbw/project/LIDC-IDRI/nodules_hires/labels_multitask.csv'

PYLIDCRC = os.path.expanduser('~/.pylidcrc')

def configure_pylidc(dicom_dir):
    with open(PYLIDCRC, 'w') as f:
        f.write(f"[dicom]\npath = {dicom_dir}\n")

# pylidc 需要指向 DICOM 路徑，但我們已刪除。
# 改用已有的 labels.csv 資訊 + pylidc 記憶中的標注讀取
# pylidc 的 Annotation 存在 SQLite，不需要 DICOM 本身

df = pd.read_csv(CSV_IN)
print(f"原始 CSV: {len(df)} 筆")

feature_map = {}  # (patient_id, nodule_idx) -> features

patients = df['patient_id'].unique()
print(f"共 {len(patients)} 個病人，開始提取特徵...")

failed = 0
for pid in patients:
    scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).all()
    if not scans:
        failed += 1
        continue
    scan = scans[0]
    try:
        clusters = scan.cluster_annotations()
    except Exception:
        failed += 1
        continue

    for nid, anns in enumerate(clusters):
        if len(anns) < 3:
            continue
        feature_map[(pid, nid)] = {
            'subtlety':   round(np.mean([a.subtlety   for a in anns]), 3),
            'sphericity': round(np.mean([a.sphericity for a in anns]), 3),
            'margin':     round(np.mean([a.margin     for a in anns]), 3),
            'lobulation': round(np.mean([a.lobulation for a in anns]), 3),
            'spiculation':round(np.mean([a.spiculation for a in anns]), 3),
            'texture':    round(np.mean([a.texture    for a in anns]), 3),
        }

print(f"提取完成: {len(feature_map)} 個結節特徵，{failed} 個病人失敗")

# 合併到 DataFrame
for col in ['subtlety','sphericity','margin','lobulation','spiculation','texture']:
    df[col] = np.nan

for (pid, nid), feats in feature_map.items():
    mask = (df['patient_id'] == pid) & (df['nodule_id'] == nid)
    for col, val in feats.items():
        df.loc[mask, col] = val

df_out = df.dropna(subset=['spiculation'])
print(f"輸出 CSV: {len(df_out)} 筆（移除無特徵的行）")
df_out.to_csv(CSV_OUT, index=False)
print(f"已儲存至 {CSV_OUT}")
print(df_out[['subtlety','sphericity','margin','lobulation','spiculation','texture','label']].describe())
