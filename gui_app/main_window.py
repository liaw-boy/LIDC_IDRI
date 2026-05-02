# gui_app/main_window.py
"""Main application window for the Lung Nodule Detection & Classification system.
Provides a premium, modern medical dashboard aesthetic with high visual fidelity.
"""

import os
import sys
import torch  # Import torch first to avoid DLL conflicts with PyQt5
import yaml
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QTabWidget,
    QPushButton,
    QLabel,
    QFileDialog,
    QMessageBox,
    QComboBox,
    QGroupBox,
    QFormLayout,
    QFrame,
    QStatusBar,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QIcon
from .image_viewer import ImageViewer
from .lung_rads_card import LungRadsPanel
from .model_manager import ModelManager
from .predictor import Predictor


class WorkerThread(QThread):
    """Background thread for running detection / classification without freezing UI."""
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            self.progress.emit("分析中...")
            result = self.func(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lung Nodule AI Diagnostic System v2.0")
        self.resize(1280, 850)
        # 確保中文字型在任何啟動方式下都生效
        app = QApplication.instance()
        if app:
            app.setFont(QFont("Noto Sans CJK TC", 13))
        
        # Load configuration
        self.config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        self.config = self._load_config()
        self.model_manager = ModelManager(self.config)
        self.predictor = Predictor(self.model_manager)

        self._setup_ui()
        self._load_stylesheet()

    def _load_stylesheet(self):
        qss_path = os.path.join(os.path.dirname(__file__), "styles.qss")
        if os.path.exists(qss_path):
            with open(qss_path, "r", encoding="utf-8") as f:
                self.setStyleSheet(f.read())
        else:
            # Fallback if file missing
            self.setStyleSheet("QMainWindow { background-color: #020617; color: white; }")

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ---------------------------------------------------------
        # SIDEBAR (STITCH DNA: SLATE 900)
        # ---------------------------------------------------------
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        title = QLabel("STITCH AI")
        title.setObjectName("sidebar_title")
        sidebar_layout.addWidget(title)

        nav_container = QWidget()
        nav_layout = QVBoxLayout(nav_container)
        nav_layout.setContentsMargins(20, 30, 20, 30)
        nav_layout.setSpacing(8)

        nav_layout.addWidget(QLabel("SOURCE DATA", objectName="section_header"))
        self.load_btn = QPushButton("📁  Import Study Folder")
        self.load_btn.clicked.connect(self._load_dicom)
        nav_layout.addWidget(self.load_btn)

        nav_layout.addWidget(QLabel("DIAGNOSTIC PIPELINE", objectName="section_header"))
        self.detect_btn = QPushButton("🎯  Launch Detection")
        self.detect_btn.setObjectName("primary_btn")
        self.detect_btn.setEnabled(False)
        self.detect_btn.clicked.connect(self._run_detection)
        nav_layout.addWidget(self.detect_btn)

        self.classify_btn = QPushButton("🧠  Run Classification")
        self.classify_btn.setObjectName("primary_btn")
        self.classify_btn.setEnabled(False)
        self.classify_btn.clicked.connect(self._run_classification)
        nav_layout.addWidget(self.classify_btn)

        self.report_btn = QPushButton("📄  Generate Report")
        self.report_btn.setEnabled(False)
        self.report_btn.clicked.connect(self._generate_report)
        nav_layout.addWidget(self.report_btn)

        nav_layout.addWidget(QLabel("INFRASTRUCTURE", objectName="section_header"))
        self.device_combo = QComboBox()
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        self.device_combo.addItems(devices)
        self.device_combo.setCurrentText(self.config.get("device", "cpu"))
        self.device_combo.currentTextChanged.connect(self._change_device)
        nav_layout.addWidget(self.device_combo)

        nav_layout.addStretch()
        
        system_status = QLabel("● ENGINE READY")
        system_status.setStyleSheet("color: #10b981; font-weight: 700; font-size: 11px; padding: 20px;")
        nav_layout.addWidget(system_status)

        sidebar_layout.addWidget(nav_container)
        layout.addWidget(sidebar)

        # ---------------------------------------------------------
        # MAIN DASHBOARD (STITCH DNA: SLATE 950)
        # ---------------------------------------------------------
        main_container = QWidget()
        main_container.setObjectName("main_container")
        main_layout = QVBoxLayout(main_container)
        main_layout.setSpacing(20)
        
        dashboard_splitter = QSplitter(Qt.Vertical)
        dashboard_splitter.setHandleWidth(2)
        
        # 1. Image Viewport Card
        viewport_card = QFrame()
        viewport_card.setObjectName("card")
        vp_layout = QVBoxLayout(viewport_card)
        vp_layout.setContentsMargins(15, 15, 15, 15)
        
        vp_header = QLabel("SURGICAL VIEWPORT")
        vp_header.setStyleSheet("color: #64748b; font-weight: 700; font-size: 11px;")
        vp_layout.addWidget(vp_header)
        
        self.viewer = ImageViewer()
        self.viewer.setStyleSheet("background-color: #000; border-radius: 8px;")
        vp_layout.addWidget(self.viewer)
        dashboard_splitter.addWidget(viewport_card)
        
        # 2. Analytics Card
        analytics_card = QFrame()
        analytics_card.setObjectName("card")
        ana_layout = QHBoxLayout(analytics_card)
        ana_layout.setContentsMargins(30, 30, 30, 30)
        ana_layout.setSpacing(40)
        
        # Result Column
        res_col = QWidget()
        res_layout = QVBoxLayout(res_col)
        res_layout.setContentsMargins(0, 0, 0, 0)
        
        res_title = QLabel("ANALYTIC INSIGHTS")
        res_title.setObjectName("result_title")
        res_layout.addWidget(res_title)
        
        self.result_label = QLabel("Initialize DICOM series to begin AI-driven nodule characterization...")
        self.result_label.setObjectName("result_text")
        self.result_label.setWordWrap(True)
        self.result_label.setAlignment(Qt.AlignTop)
        res_layout.addWidget(self.result_label)

        # Lung-RADS color-coded card panel (demo-grade visualization)
        self.lung_rads_panel = LungRadsPanel()
        res_layout.addWidget(self.lung_rads_panel, 1)
        
        ana_layout.addWidget(res_col, 3)
        
        # Visualization Column (Chart + Attention Map)
        vis_col = QWidget()
        vis_layout = QVBoxLayout(vis_col)
        
        self.chart_label = QLabel()
        self.chart_label.setAlignment(Qt.AlignCenter)
        vis_layout.addWidget(self.chart_label)
        
        # New: Attention Map Area
        attn_header = QLabel("ROI 放大圖（32×32）")
        attn_header.setStyleSheet("color: #64748b; font-weight: 700; font-size: 11px; margin-top: 10px;")
        vis_layout.addWidget(attn_header)

        self.attn_map_label = QLabel("執行分類後將顯示結節 ROI 放大圖")
        self.attn_map_label.setStyleSheet("color: #475569; border: 1px dashed #334155; border-radius: 6px; padding: 8px;")
        self.attn_map_label.setAlignment(Qt.AlignCenter)
        self.attn_map_label.setMinimumHeight(120)
        vis_layout.addWidget(self.attn_map_label)
        
        ana_layout.addWidget(vis_col, 2)
        
        dashboard_splitter.addWidget(analytics_card)
        
        # Ratios: Viewport 65%, Analytics 35%
        dashboard_splitter.setStretchFactor(0, 65)
        dashboard_splitter.setStretchFactor(1, 35)
        
        main_layout.addWidget(dashboard_splitter)
        layout.addWidget(main_container, 1)

        self.statusBar().showMessage("Stitch AI Engine Initialized")

    # ---------------------------------------------------------------------
    # Configuration helpers
    # ---------------------------------------------------------------------
    def _load_config(self):
        if not os.path.exists(self.config_path):
            base_dir = os.path.dirname(__file__)
            default = {
                "yolo": {"model_path": os.path.abspath(os.path.join(base_dir, "../models/best.pt"))},
                "cnn": {"model_path": os.path.abspath(os.path.join(base_dir, "../models/dual_input_best_auc_model.pth"))},
                "device": "cpu",
                "threshold": 0.4,
            }
            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(default, f, allow_unicode=True)
            return default
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _save_config(self):
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.config, f, allow_unicode=True)

    def _change_yolo_model(self, idx):
        self.model_manager.reload_yolo()

    def _change_cnn_model(self, idx):
        self.model_manager.reload_cnn()

    def _change_device(self, device):
        self.config["device"] = device
        self._save_config()
        self.model_manager.set_device(device)

    # ---------------------------------------------------------------------
    # File handling & core actions
    # ---------------------------------------------------------------------
    def _load_dicom(self):
        # Doctor flow: pick a single study folder; we auto-scan all medical images inside
        folder = QFileDialog.getExistingDirectory(self, "選擇病患 study 資料夾")
        if not folder:
            return
        files = self._scan_study_folder(folder)
        if not files:
            QMessageBox.warning(self, "載入失敗",
                                "資料夾內找不到 DICOM (.dcm) 或影像 (.png/.jpg) 檔案。")
            return
        self.dicom_paths = files
        pixmap = self.viewer.load_dicom_series(files)
        if pixmap:
            self.viewer.setPixmap(pixmap)
            self.detect_btn.setEnabled(True)
            self.result_label.setText(
                f"已載入 {len(files)} 片影像。\n"
                f"Study folder: {os.path.basename(folder)}\n"
                f"系統準備就緒，請啟動偵測程序。"
            )
            self.statusBar().showMessage(f"Loaded {len(files)} slices from {folder}")
        else:
            QMessageBox.warning(self, "載入失敗", "無法解析選定的影像檔案。")

    def _scan_study_folder(self, folder: str) -> list[str]:
        """Recursively find medical image files in a study folder, sorted."""
        exts = (".dcm", ".png", ".jpg", ".jpeg")
        found: list[str] = []
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(exts):
                    found.append(os.path.join(root, f))
        return sorted(found)

    def _run_detection(self):
        if not hasattr(self, "dicom_paths"):
            return
        self._run_in_thread(self.predictor.run_detection, self.dicom_paths)

    def _run_classification(self):
        if not hasattr(self, "dicom_paths"):
            return
        self._run_in_thread(self.predictor.run_classification, self.dicom_paths)

    def _run_in_thread(self, func, *args):
        if hasattr(self, "thread") and self.thread.isRunning():
            QMessageBox.warning(self, "請稍候", "上一個分析任務仍在執行中，請等待完成後再試。")
            return
        self.detect_btn.setEnabled(False)
        self.classify_btn.setEnabled(False)
        self.thread = WorkerThread(func, *args)
        self.thread.progress.connect(lambda msg: self.statusBar().showMessage(msg))
        self.thread.finished.connect(self._handle_result)
        self.thread.finished.connect(lambda: self.detect_btn.setEnabled(True))
        self.thread.error.connect(lambda err: QMessageBox.critical(self, "運算錯誤", err))
        self.thread.error.connect(lambda: self.detect_btn.setEnabled(True))
        self.thread.start()

    def _handle_result(self, result):
        self.statusBar().clearMessage()
        self.result_label.setText(result.get("message", ""))
        if "chart" in result:
            from PyQt5.QtGui import QPixmap
            import io
            img_bytes = io.BytesIO(result["chart"])
            pix = QPixmap()
            pix.loadFromData(img_bytes.read())
            self.chart_label.setPixmap(pix.scaled(450, 320, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        if result.get("type") == "detection":
            self.classify_btn.setEnabled(True)
            self.statusBar().showMessage("Detection completed")
            if "annotated_image" in result:
                self.viewer.set_pixmap_from_bytes(result["annotated_image"])
        elif result.get("type") == "classification":
            self.report_btn.setEnabled(True)
            self.statusBar().showMessage("Analysis completed")
            if "nodules" in result and hasattr(self, "lung_rads_panel"):
                self.lung_rads_panel.render_nodules(result["nodules"])
            if result.get("ct_with_attention"):
                self.viewer.set_pixmap_from_bytes(result["ct_with_attention"])
            if "attention_map" in result:
                pix = QPixmap()
                pix.loadFromData(result["attention_map"])
                self.attn_map_label.setPixmap(pix.scaled(400, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _generate_report(self):
        self.statusBar().showMessage("Generating PDF Report...")
        try:
            from .pdf_report import generate_pdf_report
            import datetime
            nodules = list(getattr(self.predictor, "_last_nodules", []) or [])
            if not nodules:
                QMessageBox.warning(self, "報告生成", "尚未執行分類，無法生成報告。")
                self.statusBar().clearMessage()
                return
            patient_id = "Unknown"
            if getattr(self, "dicom_paths", None):
                base = os.path.basename(os.path.dirname(self.dicom_paths[0]))
                if base:
                    patient_id = base
            output_dir = self.config.get("output_dir", "output")
            os.makedirs(output_dir, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_path = os.path.join(output_dir, f"report_{patient_id}_{ts}.pdf")
            kpi = {"n_test": 70, "recall": 1.0, "fpr": 0.054, "f1": 0.971}
            shot_path = os.path.join(output_dir, f"snapshot_{patient_id}_{ts}.png")
            self.grab().save(shot_path)
            generate_pdf_report(pdf_path, patient_id, nodules,
                                screenshots=(shot_path,), kpi=kpi)
            QMessageBox.information(self, "報告生成", f"診斷報告已生成: {pdf_path}")
            self.statusBar().showMessage(f"Report saved: {pdf_path}")
        except Exception as e:
            QMessageBox.critical(self, "報告生成失敗", str(e))
            self.statusBar().clearMessage()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
