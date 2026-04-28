# 肺結節良惡性輔助診斷系統 (Lung Nodule Detection & Malignancy Classification)

本專案為淡江大學資訊工程系專題研究成果，開發一套結合深度學習物件偵測與多尺度特徵提取分類技術的醫療影像輔助系統。

## 🌟 系統特點

- **高精度偵測**：採用 **YOLOv11** 物件偵測演算法，針對 LUNA16 資料集進行優化，能精準定位 CT 影像中的肺結節位置。
- **雙路徑特徵融合 (Dual Input CNN)**：
    - **ROI 路徑**：聚焦結節局部型態特徵。
    - **Full CT 路徑**：捕捉結節周圍組織與全局上下文資訊。
    - 結合兩者特徵，顯著提升良惡性判斷的準確率（達 94.7%），並降低誤診率。
- **整合式 GUI 介面**：基於 **PyQt5** 開發，提供醫師直觀的作業環境，支援 DICOM 影像載入、自動偵測與良惡性結果標註。

## 📂 專案結構

- **/gui_app**: 系統主程式 (`cnn_detector_v1.py`) 與介面邏輯。
- **/detection_yolo**: YOLOv11 訓練腳本與偵測模型配置。
- **/classification_cnn**: 雙輸入分類模型架構、訓練程式與 DataLoader。
- **/preprocessing**: LIDC-IDRI/LUNA16 原始數據處理與 COCO 格式轉換工具。
- **/models**: (建議存放處) 存放訓練好的 `.pt` 與 `.pth` 模型權重（未包含在 Git 中）。

## 🚀 執行環境

### 必要套件
```bash
pip install ultralytics pydicom PyQt5 torch torchvision opencv-python numpy
```

### 啟動系統
1. 請確保將訓練好的 `best.pt` 與 `dual_input_best_auc_model.pth` 放入 `gui_app/` 目錄或指定路徑。
2. 執行主程式：
```bash
python gui_app/cnn_detector_v1.py
```

## 📊 研究成果
- **偵測準確率 (mAP@0.5)**: 0.88+
- **分類準確率 (Accuracy)**: 94.7%
- **分類召回率 (Recall)**: 94.7%

## 👥 研究團隊
- **指導老師**：淡江大學資訊工程系 教授
- **組員**：陳威丞、廖柏維、鍾翔宇、江昊宸

---
*本系統僅供研究與輔助參考用途，臨床診斷請以專業醫師判斷為準。*
