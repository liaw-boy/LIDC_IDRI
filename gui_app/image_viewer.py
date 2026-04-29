# gui_app/image_viewer.py
"""A simple image viewer widget based on QGraphicsView.
Supports zoom with mouse wheel and drag‑pan.
It can load a single DICOM file or a series of DICOM files (as a stack).
"""

import os
import numpy as np
import pydicom
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap, QImage, QPainter
from PyQt5.QtCore import Qt


class ImageViewer(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene().addItem(self.pixmap_item)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self._zoom = 0

    def setPixmap(self, pixmap: QPixmap):
        self.pixmap_item.setPixmap(pixmap)
        self._reset_view()

    # -----------------------------------------------------------------
    # DICOM handling – load the first slice as a preview image
    # -----------------------------------------------------------------
    def load_dicom_series(self, file_paths):
        """Load a series of DICOM files, return QPixmap of the first slice.
        The full list is stored for later processing.
        """
        if not file_paths:
            return None
        # Load first slice to display
        ds = pydicom.dcmread(file_paths[0])
        img = ds.pixel_array
        if img.ndim == 3:  # sometimes colour
            img = img[..., 0]
        # Normalise to 0‑255
        img = ((img - img.min()) / (img.ptp() + 1e-8) * 255).astype(np.uint8)
        height, width = img.shape
        qimg = QImage(img.data, width, height, width, QImage.Format_Grayscale8)
        pix = QPixmap.fromImage(qimg)
        # Store paths for later use
        self.dicom_paths = file_paths
        return pix

    # -----------------------------------------------------------------
    # Zoom handling
    # -----------------------------------------------------------------
    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            angle = event.angleDelta().y()
            factor = 1.25 if angle > 0 else 0.8
            self.scale(factor, factor)
            self._zoom += 1 if angle > 0 else -1
        else:
            super().wheelEvent(event)

    def _reset_view(self):
        self.resetTransform()
        self._zoom = 0
        self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
