# gui_app/cnn_detector_v1.py
"""Entry point for the Lung Nodule Detection & Classification system.
Launches the modern MainWindow.
"""

import sys
import torch # Import torch at the very top to avoid DLL conflicts on Windows
from PyQt5.QtWidgets import QApplication
from .main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
