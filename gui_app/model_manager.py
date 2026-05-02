# gui_app/model_manager.py
"""Utility class to load and manage YOLO and CNN models.
All heavy model objects are lazy‑loaded and kept on the selected device.
Configuration is read from a yaml file (config.yaml) located in the same folder.
"""

import os
import yaml
import torch
from ultralytics import YOLO
from .nodule_classifier import NoduleClassifier


class ModelManager:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device(config.get("device", "cpu"))
        self.yolo_model = None
        self.cnn_model = None
        self._load_models()

    # ---------------------------------------------------------------------
    # Loading helpers
    # ---------------------------------------------------------------------
    def _load_models(self):
        self.reload_yolo()
        self.reload_cnn()

    def reload_yolo(self):
        yolo_path = self.config["yolo"]["model_path"]
        if not os.path.isabs(yolo_path):
            yolo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", yolo_path))
        if not os.path.exists(yolo_path):
            raise FileNotFoundError(f"YOLO model not found at {yolo_path}")
        self.yolo_model = YOLO(yolo_path)
        # ultralytics handles device internally; we set it here for consistency
        self.yolo_model.to(self.device)

    def reload_cnn(self):
        cnn_path = self.config["cnn"]["model_path"]
        if not os.path.isabs(cnn_path):
            cnn_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", cnn_path))
        if not os.path.exists(cnn_path):
            raise FileNotFoundError(f"CNN model not found at {cnn_path}")

        raw_sd = torch.load(cnn_path, map_location=self.device, weights_only=True)
        # Detect Attribute Feedback wrapper checkpoint
        # (saved by train_attfeedback.py — keys prefixed `backbone.` plus aux_head/malignancy_head)
        has_attfb = any(k.startswith("aux_head.") or k.startswith("malignancy_head.")
                        for k in raw_sd.keys())
        if has_attfb:
            sd = {(k[len("backbone."):] if k.startswith("backbone.") else k): v
                  for k, v in raw_sd.items()}
            n_aux_w = raw_sd.get("aux_head.2.weight")
            n_aux = n_aux_w.shape[0] if n_aux_w is not None else 3
            self.cnn_model = NoduleClassifier(use_attribute_feedback=True, n_aux=n_aux)
        else:
            sd = raw_sd
            self.cnn_model = NoduleClassifier(use_attribute_feedback=False)

        self.cnn_model.load_state_dict(sd)
        self.cnn_model.eval()
        self.cnn_model.to(self.device)
        print(f"[CNN] loaded {'AttributeFeedback' if has_attfb else 'baseline'} model from {os.path.basename(cnn_path)}")

    def get_yolo(self):
        return self.yolo_model

    def get_cnn(self):
        return self.cnn_model

    # ---------------------------------------------------------------------
    # Device management
    # ---------------------------------------------------------------------
    def set_device(self, device_str: str):
        self.device = torch.device(device_str)
        if self.yolo_model:
            self.yolo_model.to(self.device)
        if self.cnn_model:
            self.cnn_model.to(self.device)
        self.config["device"] = device_str
        # Save back to yaml for persistence
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.config, f, allow_unicode=True)

    # ---------------------------------------------------------------------
    # Prediction interfaces used by Predictor
    # ---------------------------------------------------------------------
    def detect(self, image_np):
        """Run YOLO detection on a NumPy image (H, W, C).
        Returns list of bounding boxes (x1, y1, x2, y2, confidence, class).
        """
        # ultralytics expects BGR format; ensure correct ordering
        results = self.yolo_model(image_np, save=False, device=self.device)
        # each result contains .boxes with xyxy and conf
        detections = []
        for r in results:
            boxes = r.boxes
            for b in boxes:
                xyxy = b.xyxy.cpu().numpy().flatten().tolist()
                conf = float(b.conf.cpu().numpy())
                cls = int(b.cls.cpu().numpy())
                detections.append((*xyxy, conf, cls))
        return detections

    def classify(self, roi_tensor, full_ct_tensor=None):
        """Run the dual‑input CNN classifier.
        roi_tensor / full_ct_tensor are torch tensors already on the correct device.
        Returns (prediction, confidence).
        """
        with torch.no_grad():
            outputs, _ = self.cnn_model(roi_tensor, full_ct_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, dim=1)
            return int(pred.item()), float(confidence.item())
