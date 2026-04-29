# 肺結節良惡性輔助診斷系統 (Lung Nodule Detection & Malignancy Classification)

本專案為淡江大學資訊工程系專題研究成果，開發一套結合深度學習物件偵測與多尺度特徵提取分類技術的醫療影像輔助系統。

## 📊 系統架構與流程

### 專題流程圖
![專題流程圖](docs/images/image1.png)

### 雙路徑特徵融合模型 (Dual Input CNN)
模型同時分析結節局部 (ROI) 與全局上下文 (Full CT) 資訊，以達到更高精度的分類。
![模型架構圖](docs/images/image6.png)
*(圖：Full CT 路徑與 ROI 路徑示意圖)*

## 🖥️ 系統介面呈現
![GUI 介面](docs/images/image20.png)
*(系統可自動標註結節位置，並即時預測良惡性程度與信心度)*

## 📈 模型效能評估
本研究對比了「簡易 CNN」與「雙輸入 CNN」的表現，結果顯示雙輸入架構在各項指標上均有顯著提升：

| 評估指標 | 簡易 CNN 模型 | 雙輸入模型 (本專案) | 提升幅度 |
| :--- | :--- | :--- | :--- |
| **準確率 (Accuracy)** | 84.30% | **94.70%** | **+10.40%** |
| **召回率 (Recall)** | 89.52% | **94.70%** | +5.18% |
| **特異度 (Specificity)** | 78.61% | **94.60%** | +15.99% |
| **AUC 值** | 0.919 | **0.984** | +7.07% |
| **假陽性 (FP)** | 172 例 | **38 例** | **-77.90%** |
| **假陰性 (FN)** | 92 例 | **50 例** | -45.70% |

---

## 📂 專案結構 (Project Structure)

本專案採模組化設計，各目錄功能說明如下：

- **`gui_app/`**: 系統主程式與 UI 組件 (PyQt5)。
- **`models/`**: 存放 YOLOv11 與 3D CNN 的模型權重 (`.pt`, `.pth`)。
- **`preprocessing/`**: 醫療影像預處理腳本 (DICOM 標註、特徵提取)。
- **`classification_cnn/`**: CNN 模型訓練、Dataloader 與模型架構定義。
- **`detection_yolo/`**: YOLO 偵測模型訓練配置與腳本。
- **`scripts/`**: 通用工具腳本（如標註 CSV 整合、檔案移動等）。
- **`data/`**: 存放標籤數據與測試樣本 (`data/sample`)。
- **`docs/`**: 專案文檔、架構圖與開發手冊。
- **`output/`**: 預設的診斷結果輸出目錄。

---

## 🚀 執行環境

### 必要套件
```bash
pip install ultralytics pydicom PyQt5 torch torchvision opencv-python numpy
```

### 啟動系統
1. **快速設定 (Windows)**：執行專案根目錄下的 `setup.bat` 即可自動建立虛擬環境並安裝依賴。
2. **手動啟動**：
   - 請確保將訓練好的 `best.pt` 與 `dual_input_final_model.pth` 放入 `models/` 目錄。
   - 執行主程式：
     ```bash
     python -m gui_app.cnn_detector_v1
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
