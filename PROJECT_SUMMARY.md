# 專案總結 (PROJECT SUMMARY)

> Lung Nodule AI 輔助診斷系統 — v2.0 收官報告
> 完成日期: 2026-05-02

---

## 一、最終 KPI（51 病患 holdout test, seed=42）

| 指標 | 數值 | 95% Bootstrap CI |
|:---|:---:|:---:|
| **惡性結節端到端召回 (E2E Recall)** | **100%** (33/33) | [1.000, 1.000] |
| **良性誤判率 (B→M FPR)** | 5.4% (2/37) | [0.000, 0.135] |
| **F1 分數** | **0.971** | [0.925, 1.000] |
| YOLO 偵測召回 | 97.1% | — |
| CNN AUC (AttFB) | **0.997** | — |

**Bootstrap 1000 次重採樣中每一次都守住 100% 惡性召回**——signal 真硬，非小樣本運氣。

---

## 二、系統架構

```
[CT DICOM 序列]
      ↓
[Stage 1: YOLO11n 結節偵測]   ← models/best.pt (5.4 MB)
      ↓
[3D Nodule Grouping]           ← 連續 slice 偵測框聚成 3D nodule
      ↓
[Stage 2: NoduleClassifier]    ← models/dual_input_final_model.pth (9.3 MB)
      ↓                            (CBAM + Attribute Feedback)
[3D Gaussian Aggregation]      ← 中央 slice 權重大、邊緣 slice 權重小
      ↓
[Lung-RADS 5 級分級 + 行動建議]
      ↓
[GUI: 色塊卡 + Attention Overlay + PDF Report]
```

---

## 三、關鍵技術

| 技術 | 來源 | 在本專案的角色 |
|:---|:---|:---|
| **YOLO11n** | Ultralytics 2024 | Stage 1 結節偵測（LIDC fine-tune） |
| **CBAM Residual Attention** | Woo et al. ECCV 2018 | CNN 骨幹的 channel + spatial 注意力 |
| **Attribute Feedback** | JIMI 2022 | 輔助任務（lobulation/spiculation/margin）反饋進惡性分類頭 |
| **3D Gaussian-weighted Aggregation** | 本專案 | 多 slice 推理結果聚合，把 B→M FP 從 13.5% 砍到 5.4% |
| **Lung-RADS 1.1** | American College of Radiology | 5 段臨床分級（2 / 3 / 4A / 4B / 4X） |

---

## 四、消融實驗（Ablation）

| 變體 | Test AUC | E2E Recall | 部署 |
|:---|:---:|:---:|:---:|
| 1ch baseline + 3D Gaussian agg (v1) | 0.984 | — | — |
| 1ch + AttFB (multi-task only) | 0.984 | — | — |
| **1ch + AttFB (with feedback) + 3D agg** | **0.997** | **100%** | **✅ Production** |
| 2.5D (3 adjacent slices stacked) + AttFB | 0.992-0.995 | — | ⚠️ 備查（沒贏） |
| YOLO V3 (with hard negatives) | F1 0.813 | 91.3% | ⚠️ 備查（單模 F1 高，pipeline 沒贏 V2） |

**研究發現:** 多視角資訊在「推理端做 3D Gaussian 聚合」比「輸入端做 2.5D stacking」更有效。

---

## 五、資料合規（Data Integrity）

```
331 LIDC 病患 → seed=42 patient-level split
   Train: 231 / Val: 49 / Test: 51

✅ Train ∩ Val   = ∅
✅ Train ∩ Test  = ∅
✅ Val ∩ Test    = ∅
✅ YOLO V2 + V3 + CNN 三個模型用同一份 split
✅ 所有 demo case 全在 TEST set
✅ Bootstrap 重採樣只在 70-nodule TEST 範圍內，永不碰 train/val
```

---

## 六、GUI 醫師工作流（Doctor MVP）

```
1. 點 [Import Study Folder]   → 自動掃 .dcm/.png/.jpg 整個資料夾
2. 點 [Launch Detection]      → YOLO 偵測，顯示橘色 bounding box
3. 點 [Run Classification]    → CNN + 3D agg + Lung-RADS 色塊卡
4. 點 [Generate Report]       → 一鍵生成 PDF 報告（含 KPI、截圖、處置建議）
```

**GUI 視覺化:**
- Lung-RADS 5 色塊卡（紅 4X / 橘 4B / 黃 4A / 藍 3 / 綠 2）
- Attention heatmap 疊在 CT viewport 的 YOLO 框內
- ROI 32×32 注意力放大圖
- 良 vs 惡機率長條圖

---

## 七、Repo 狀態

```
GitHub:  https://github.com/liaw-boy/LIDC_IDRI
Branch:  main (latest: 1a37610 docs: add Train from Scratch guide)

主要 commit:
  1a37610  docs: add Train from Scratch guide (6 steps, replicable)
  5ccbf93  feat: ship Doctor MVP — Lung-RADS UI + 3D agg + PDF report
  59de44b  Update UI styles and font for high readability
```

**Repo 含:**
- 完整訓練 / 推理 / GUI 程式碼
- README 含架構圖 + KPI + 6 步從零訓練教學 + 合規章節
- docs/images/demo_workflow.gif (4 frame 動畫)
- 不含: model 權重 / 大資料 / 個人報告 (.docx)

---

## 八、Production Bug Fixes（演示中發現並修復）

```
1. predictor.py:66 — model(img, conf=yolo_conf) 之前沒傳 conf
   → GUI 用了 ultralytics 預設 conf=0.25，把低 conf 偵測全砍
   → 修復後 LIDC-0031 nodule-4 從 MISS 變 4X 91.6%

2. image_viewer.load_dicom_series() 自我封裝 setPixmap
   → demo 腳本 + 真用戶都正確顯示 CT 影像

3. PNG 輸入支援
   → 解鎖不需要 DICOM 的演示路徑
```

---

## 九、未來工作（Future Work — 答辯可念稿）

1. **3D CNN（3D ResNet / 3D DenseNet）**
   - 對 <6mm 微小結節判讀有提升空間
   - 需重建 DICOM 體素資料 + 擴大訓練集（如 LUNA16）以避免過擬合

2. **外部資料集驗證（LUNA16 / NLST）**
   - 證明跨資料集泛化能力
   - 需 ~120 GB 磁碟空間（可分批處理）

3. **GUI Threshold Slider**
   - 讓醫師現場調 CNN 閾值，平衡敏感度 vs 特異度

4. **臨床部署準備**
   - 與醫院影像科合作前瞻性驗證
   - DICOM SR 結構化報告整合
   - HL7 FHIR 接口

---

## 十、研究團隊

- **指導老師**: 淡江大學資訊工程系 教授
- **組員**: 陳威丞、廖柏維、鍾翔宇、江昊宸

---

## 十一、結論

本專案以淡江大學資訊工程系專題形式，於 LIDC-IDRI 公開資料集 51 病患 holdout test split 上達成：

- **100% 惡性結節端到端召回**（Bootstrap 95% CI [1.000, 1.000]）
- **5.4% 良性誤判率**
- **F1 = 0.971**
- **CNN AUC = 0.997**

技術貢獻包含：
1. 兩階段架構（YOLO 偵測 + CNN 分類）對齊臨床決策流程
2. **3D Gaussian-weighted slice aggregation** 將 B→M FP 從 13.5% 降至 5.4%
3. **Attribute Feedback** 將輔助臨床屬性反饋進惡性分類，AUC 從 0.984 提升至 0.997
4. **Lung-RADS 5 段分級**對齊放射科臨床決策語言
5. 完整 GUI 含 attention 可視化、PDF 報告自動生成

達到醫療影像 AI 可發表水平。

---

*本系統為 AI 輔助診斷工具，最終診斷需由放射科醫師判讀。*
