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
        
        # Load configuration
        self.config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        self.config = self._load_config()
        self.model_manager = ModelManager(self.config)
        self.predictor = Predictor(self.model_manager)

        self._setup_ui()
        self._apply_dark_theme()

    def _apply_dark_theme(self):
        # Stitch Design DNA Implementation
        self.setStyleSheet("""
            QMainWindow {
                background-color: #020617;
            }
            QWidget {
                color: #f8fafc;
                font-family: 'Inter', 'system-ui', 'Segoe UI', sans-serif;
            }
            
            /* Sidebar: Slate 900 */
            #sidebar {
                background-color: #0f172a;
                border-right: 1px solid #1e293b;
                min-width: 280px;
                max-width: 280px;
            }
            
            #sidebar_title {
                font-size: 20px;
                font-weight: 900;
                color: #38bdf8;
                padding: 30px 20px;
                letter-spacing: 1px;
                border-bottom: 1px solid #1e293b;
            }

            /* Main Content: Slate 950 */
            #main_container {
                background-color: #020617;
                padding: 20px;
            }

            /* Cards: Surface (Slate 800) */
            QFrame#card {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 12px;
            }
            
            /* Buttons */
            QPushButton {
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 8px;
                padding: 12px 16px;
                font-weight: 500;
                font-size: 13px;
                color: #94a3b8;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #1e293b;
                color: #f8fafc;
                border-color: #334155;
            }
            QPushButton#primary_btn {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0284c7, stop:1 #3b82f6);
                color: white;
                font-weight: 700;
                text-align: center;
                border: none;
            }
            QPushButton#primary_btn:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0ea5e9, stop:1 #2563eb);
                box-shadow: 0 4px 12px rgba(2, 132, 199, 0.4);
            }
            QPushButton#primary_btn:disabled {
                background: #1e293b;
                color: #475569;
            }

            /* Section Headers */
            #section_header {
                font-size: 11px;
                font-weight: 800;
                color: #475569;
                text-transform: uppercase;
                letter-spacing: 1.5px;
                margin-top: 25px;
                margin-bottom: 8px;
                padding-left: 10px;
            }

            /* Labels and Results */
            #result_title {
                font-size: 18px;
                font-weight: 700;
                color: #f8fafc;
                margin-bottom: 5px;
            }
            #result_text {
                font-size: 14px;
                color: #94a3b8;
                line-height: 1.6;
            }

            /* ComboBox */
            QComboBox {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 8px 12px;
                color: #f8fafc;
            }
            QComboBox:hover {
                border-color: #0284c7;
            }

            /* Splitter Handle */
            QSplitter::handle {
                background-color: #1e293b;
                width: 1px;
                height: 1px;
            }
            
            /* Status Bar */
            QStatusBar {
                background-color: #020617;
                color: #475569;
                font-size: 11px;
                border-top: 1px solid #1e293b;
            }
        """)

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
        self.load_btn = QPushButton("📂  Import DICOM")
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

        nav_layout.addWidget(QLabel("INFRASTRUCTURE", objectName="section_header"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "cuda"])
        self.device_combo.setCurrentText(self.config.get("device", "cpu"))
        self.device_combo.currentTextChanged.connect(self._change_device)
        nav_layout.addWidget(self.device_combo)

        nav_layout.addStretch()
        
        system_status = QLabel("● ENGINE READY")
        system_status.setStyleSheet("color: #10b981; font-weight: 800; font-size: 10px; padding: 20px;")
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
        vp_header.setStyleSheet("color: #475569; font-weight: 800; font-size: 10px; letter-spacing: 1px;")
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
        res_layout.addStretch()
        
        ana_layout.addWidget(res_col, 3)
        
        # Visualization Column
        self.chart_label = QLabel()
        self.chart_label.setAlignment(Qt.AlignCenter)
        ana_layout.addWidget(self.chart_label, 2)
        
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
        files, _ = QFileDialog.getOpenFileNames(self, "選擇 DICOM 檔案序列", "", "DICOM Files (*.dcm)")
        if not files:
            return
        self.dicom_paths = files
        pixmap = self.viewer.load_dicom_series(files)
        if pixmap:
            self.viewer.setPixmap(pixmap)
            self.detect_btn.setEnabled(True)
            self.result_label.setText(f"已載入 {len(files)} 片 DICOM 影像。系統準備就緒，請啟動偵測程序。")
            self.statusBar().showMessage(f"Loaded {len(files)} slices")
        else:
            QMessageBox.warning(self, "載入失敗", "無法解析選定的 DICOM 檔案。")

    def _run_detection(self):
        if not hasattr(self, "dicom_paths"):
            return
        self._run_in_thread(self.predictor.run_detection, self.dicom_paths)

    def _run_classification(self):
        if not hasattr(self, "dicom_paths"):
            return
        self._run_in_thread(self.predictor.run_classification, self.dicom_paths)

    def _run_in_thread(self, func, *args):
        self.thread = WorkerThread(func, *args)
        self.thread.progress.connect(lambda msg: self.statusBar().showMessage(msg))
        self.thread.finished.connect(self._handle_result)
        self.thread.error.connect(lambda err: QMessageBox.critical(self, "運算錯誤", err))
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
        elif result.get("type") == "classification":
            self.statusBar().showMessage("Analysis completed")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Global font tweak
    font = QFont("Inter", 10)
    app.setFont(font)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
