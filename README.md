# Lung Nodule Malignancy Auxiliary System (肺結節良惡性輔助系統)

本專案旨在利用深度學習技術（YOLOv11 與 Dual Input CNN）開發一套肺結節自動偵測與良惡性分類系統。

## 專案結構

- **/preprocessing**: 原始資料集 (LIDC-IDRI/LUNA16) 的預處理、XML 標記解析與格式轉換腳本。
- **/detection_yolo**: YOLOv11 結節偵測模型的訓練與配置。
- **/classification_cnn**: 雙輸入 (ROI + Full CT) 卷積神經網路的訓練與資料載入。
- **/gui_app**: 基於 PyQt5 開發的整合介面主程式。
- **/models**: 存放訓練好的模型權重檔 (.pt, .pth)。

## 快速開始

1. 安裝必要套件: `pip install ultralytics pydicom PyQt5 torch opencv-python tqdm`
2. 執行 GUI 程式: `python gui_app/cnn_detector_v1.py`

## 研究團隊
淡江大學資訊工程系 - 專題成果
組員：陳威丞、廖柏維、鍾翔宇、江昊宸
