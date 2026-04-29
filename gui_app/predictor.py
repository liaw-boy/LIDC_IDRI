# gui_app/predictor.py
"""Predictor module for handling YOLO detection and CNN classification logic.
Supports multi-slice integration and result visualization.
"""

import os
import cv2
import torch
import numpy as np
import pydicom
import torch.nn.functional as F
import matplotlib.pyplot as plt
import io


class Predictor:
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.detection_results = []  # Stores (filename, has_nodule, boxes, image)
        self.nodule_groups = []      # Groups of indices forming a 3D nodule

    def run_detection(self, dicom_paths):
        """Run YOLO detection on a series of DICOM files."""
        model = self.model_manager.get_yolo()
        device = self.model_manager.device
        self.detection_results = []

        total = len(dicom_paths)
        for idx, path in enumerate(dicom_paths):
            try:
                ds = pydicom.dcmread(path)
                pixel_array = ds.pixel_array
                # Normalize to 0-255
                img = ((pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                results = model(img_rgb, verbose=False)
                valid_boxes = []
                for result in results:
                    for box, score in zip(result.boxes.xyxy, result.boxes.conf):
                        x_min, y_min, x_max, y_max = map(int, box[:4])
                        valid_boxes.append((x_min, y_min, x_max, y_max, float(score)))

                self.detection_results.append({
                    "path": path,
                    "filename": os.path.basename(path),
                    "has_nodule": len(valid_boxes) > 0,
                    "boxes": valid_boxes,
                    "image": img
                })
            except Exception as e:
                print(f"Error processing {path}: {e}")

        # Grouping logic: continuous slices with nodules (at least 2)
        self.nodule_groups = []
        current_group = []
        n = len(self.detection_results)
        
        # Simple continuous group detection
        for i in range(n):
            if self.detection_results[i]["has_nodule"]:
                current_group.append(i)
            else:
                if len(current_group) >= 2:
                    self.nodule_groups.append(current_group)
                current_group = []
        if len(current_group) >= 2:
            self.nodule_groups.append(current_group)

        msg = f"偵測完成！共處理 {total} 片切片，找到 {len(self.nodule_groups)} 個疑似 3D 結節目標。"
        return {"type": "detection", "message": msg}

    def run_classification(self, dicom_paths):
        """Run CNN classification on detected nodules using multi-slice integration."""
        cnn_model = self.model_manager.get_cnn()
        device = self.model_manager.device
        threshold = self.model_manager.config.get("threshold", 0.4)

        if not self.nodule_groups:
            return {"type": "classification", "message": "未發現足夠的連續結節切片進行分類。"}

        results_summary = []
        all_nodule_probs = []

        for g_idx, group in enumerate(self.nodule_groups):
            nodule_rois = []
            nodule_full_cts = []
            
            for idx in group:
                data = self.detection_results[idx]
                img = data["image"]
                if not data["boxes"]: continue
                
                # Take the first box for now
                x1, y1, x2, y2, _ = data["boxes"][0]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Extract ROI 32x32
                r = 16
                rx1, ry1 = max(0, cx - r), max(0, cy - r)
                rx2, ry2 = min(img.shape[1], cx + r), min(img.shape[0], cy + r)
                roi = img[ry1:ry2, rx1:rx2]
                if roi.shape != (32, 32):
                    roi = cv2.resize(roi, (32, 32))
                
                # Extract Full CT 640x640 (resized)
                full_ct = cv2.resize(img, (640, 640))
                
                # Preprocess
                roi_t = torch.from_numpy(roi).float().unsqueeze(0).unsqueeze(0) / 255.0
                full_t = torch.from_numpy(full_ct).float().unsqueeze(0).unsqueeze(0) / 255.0
                
                nodule_rois.append(roi_t.to(device))
                nodule_full_cts.append(full_t.to(device))

            # Multi-slice integration: Weighted average (Gaussian)
            n_slices = len(nodule_rois)
            if n_slices == 0: continue
            
            center = (n_slices - 1) / 2
            sigma = max(n_slices / 6, 1.0)
            weights = [np.exp(-((i - center) ** 2) / (2 * sigma ** 2)) for i in range(n_slices)]
            weights = torch.tensor(weights).to(device)
            weights = weights / weights.sum()

            combined_probs = torch.zeros(2).to(device)
            with torch.no_grad():
                for i in range(n_slices):
                    outputs, _ = cnn_model(nodule_rois[i], nodule_full_cts[i])
                    probs = F.softmax(outputs, dim=1)[0]
                    combined_probs += probs * weights[i]

            final_probs = combined_probs.cpu().numpy()
            pred_class = 1 if final_probs[1] > threshold else 0
            label = "惡性" if pred_class == 1 else "良性"
            conf = final_probs[pred_class]
            
            results_summary.append(f"結節 #{g_idx+1}: {label} (信心度: {conf:.2%})")
            all_nodule_probs.append(final_probs)

        # Generate summary message
        msg = "\n".join(results_summary)
        
        # Generate Chart
        chart_bytes = self._generate_chart(all_nodule_probs)
        
        return {
            "type": "classification",
            "message": f"分類完成！\n\n{msg}",
            "chart": chart_bytes
        }

    def _generate_chart(self, all_probs):
        """Generate a bar chart of probabilities for all detected nodules."""
        if not all_probs:
            return None
            
        plt.figure(figsize=(6, 4))
        plt.style.use('dark_background')
        
        nodules = [f"Nodule {i+1}" for i in range(len(all_probs))]
        benign_probs = [p[0] for p in all_probs]
        malignant_probs = [p[1] for p in all_probs]
        
        x = np.arange(len(nodules))
        width = 0.35
        
        plt.bar(x - width/2, benign_probs, width, label='良性', color='#4CAF50')
        plt.bar(x + width/2, malignant_probs, width, label='惡性', color='#F44336')
        
        plt.ylabel('機率')
        plt.title('肺結節良惡性預測結果')
        plt.xticks(x, nodules)
        plt.legend()
        plt.ylim(0, 1.1)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        return buf.getvalue()
