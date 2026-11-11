# 還原說明（restore-original 分支）

目標：還原 `專題成果報告書` 的結果（基底 commit `eb60e3e`）。本分支不含後續優化，優化請放另一個 repo。

## 流程（`reproduce/`）
| 步驟 | 腳本 | 說明 |
|---|---|---|
| 0 | TCIA LIDC-IDRI + `LIDC-XML-only.zip` | 以 pylidc 資料庫挑切片厚度 ≤ 2.5 mm 的 CT（LUNA16 式篩選） |
| 1 | `01_build_tree.py` | 排成原腳本要的 `病人/series/{dcm, xml}` 結構（symlink） |
| 2 | `02_build_detection.py` | 執行原版 `preprocessing/ann_v1.py`（≥2 位醫生、非結節切片） |
| 3 | `03_build_classification.py` | 執行原版 `classification_cnn/dataloader.py`（良性≤2／惡性≥4，ROI 32×32 + 整張 CT） |
| 4 | `04_train_classifier.py` | 訓練最終分類器（殘差 + CBAM，ROI 與整張 CT 兩路，融合 5248→256→128→2） |
| 5 | `05_make_yolo_dataset.py` | COCO json → YOLO 格式，70/20/10 隨機切分 |
| 6 | `06_train_yolo.py` | YOLOv11x（`yolo11x.pt`, SGD, imgsz 640, patience 20） |
| 7 | `07_eval_pretrained.py` | 用原版權重在重建資料上評測，驗證資料分佈一致 |

`common.py`：把原腳本的 `find_dicom_file` 換成有快取的版本（結果相同，速度快很多）。

## 重要發現
- 原版權重的訓練輸入是 `(x/255-0.5)/0.5`（ROI 與整張 CT 皆是）。用 `x/255` 時 AUC 約 0.84，用 `[-1,1]` 時約 0.99。原版 GUI 推論用的是 `x/255`，與訓練不一致。
- 說明書的模型圖是簡化版；最終權重 `dual_input_final_model.pth` 含殘差 + CBAM 注意力（參考 `reproduce/ref/`）。
- 當年的 ultralytics 對 `.pt` 模型 + `pretrained=False` 仍會載入預訓練權重；新版會從頭訓練。本還原明確從 `yolo11x.pt` 開始。
- 切分為切片層級隨機（與說明書一致），同一病人切片會跨集合，數字偏樂觀。

## 權重（不進 git，本機未安裝 git-lfs）
- `models/best.pt` sha256 `68f02916d24cc806…`
- `models/dual_input_best_auc_model.pth` sha256 `0cba2d471abb9ec3…`
- `models/dual_input_final_model.pth` sha256 `957c175b30b84146…`

## 結果
YOLO 偵測的還原結果（YOLO11x，350 epoch 設定，第 292 epoch 由 patience=20 提早停止，最佳為第 281 epoch；測試集 1,785 張）：

| 測試集範圍 | 模型 | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| 說明書 | 作者 | ≈0.90 | ≈0.80 | >0.88 | >0.70 |
| 全部 1,785 張（含 825 張無結節圖） | 本次重訓 | 0.765 | 0.811 | 0.808 | 0.469 |
| 全部 1,785 張 | 作者權重 `best.pt` | 0.857 | 0.861 | 0.873 | 0.592 |
| 僅有結節的 960 張 | 本次重訓 | 0.877 | 0.840 | 0.886 | 0.511 |
| 僅有結節的 960 張 | 作者權重 `best.pt` | 0.937 | 0.899 | 0.938 | 0.634 |

說明：
- 召回率與 mAP50 達到說明書水準（以僅含結節圖計）；精確率略低；mAP50-95 未達 0.70，作者權重在本次資料上也只有 0.59 到 0.63，表示框標註或資料切分與原版有差異，不是重訓造成。
- 作者測試集的挑選方式未知，作者權重可能看過本次測試圖，其數字偏樂觀。
- 與原設定的差異：batch 8（原 16，顯卡 11GB），其餘依截圖設定。
- 端到端（YOLO 框接分類器）未評測。
