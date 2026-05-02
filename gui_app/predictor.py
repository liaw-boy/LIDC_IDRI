# gui_app/predictor.py
"""Predictor module for handling YOLO detection and CNN classification logic.
Supports multi-slice integration and result visualization.
"""

import os
import json
import datetime
import cv2
import torch
import numpy as np
import pydicom
import torch.nn.functional as F
import matplotlib.pyplot as plt
import io


def _lung_rads(mal_prob: float):
    """Map malignant probability to Lung-RADS category, label, and recommended action."""
    if mal_prob < 0.01:
        return "2", "良性", "常規年度追蹤"
    elif mal_prob < 0.05:
        return "3", "低度懷疑", "6 個月後 CT 追蹤"
    elif mal_prob < 0.15:
        return "4A", "中度懷疑", "3 個月後 CT 追蹤"
    elif mal_prob < 0.50:
        return "4B", "高度懷疑", "建議 PET-CT 或切片"
    else:
        return "4X", "高度惡性", "立即會診胸腔外科"


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
                if path.lower().endswith(".dcm"):
                    ds = pydicom.dcmread(path)
                    pixel_array = ds.pixel_array.astype(np.float32)
                    slope = float(getattr(ds, "RescaleSlope", 1))
                    intercept = float(getattr(ds, "RescaleIntercept", 0))
                    hu = pixel_array * slope + intercept
                    wc, ww = -600, 1500
                    lo, hi = wc - ww // 2, wc + ww // 2
                    img = np.clip(hu, lo, hi)
                    img = ((img - lo) / ww * 255).astype(np.uint8)
                else:
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        print(f"Skip unreadable {path}")
                        continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                yolo_conf = self.model_manager.config.get("yolo_conf", 0.5)
                min_box_px = self.model_manager.config.get("min_box_px", 5)
                results = model(img_rgb, conf=yolo_conf, verbose=False)
                valid_boxes = []
                for result in results:
                    for box, score in zip(result.boxes.xyxy, result.boxes.conf):
                        if float(score) < yolo_conf:
                            continue
                        x_min, y_min, x_max, y_max = map(int, box[:4])
                        # Filter out tiny boxes that are likely detection noise
                        if (x_max - x_min) < min_box_px or (y_max - y_min) < min_box_px:
                            continue
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

        # Final Visualization for Detection
        annotated_img = None
        if self.detection_results:
            # Find the slice with the most boxes for preview
            preview_idx = np.argmax([len(r["boxes"]) for r in self.detection_results])
            annotated_img = self._draw_hud_boxes(self.detection_results[preview_idx])

        # Grouping with gap tolerance: allow up to group_gap blank slices within a group
        gap_limit = self.model_manager.config.get("group_gap", 1)
        self.nodule_groups = []
        current_group = []
        gap_count = 0
        n = len(self.detection_results)

        for i in range(n):
            if self.detection_results[i]["has_nodule"]:
                current_group.append(i)
                gap_count = 0
            else:
                if current_group and gap_count < gap_limit:
                    gap_count += 1   # tolerate this blank slice
                else:
                    if len(current_group) >= 2:
                        self.nodule_groups.append(current_group)
                    current_group = []
                    gap_count = 0
        if len(current_group) >= 2:
            self.nodule_groups.append(current_group)

        msg = f"偵測完成！共處理 {total} 片切片，找到 {len(self.nodule_groups)} 個疑似 3D 結節目標。"
        return {
            "type": "detection", 
            "message": msg,
            "annotated_image": annotated_img
        }

    def _reset_nodule_payload(self):
        self._last_nodules = []

    def run_classification(self, dicom_paths):
        """Run CNN classification on detected nodules using multi-slice integration."""
        cnn_model = self.model_manager.get_cnn()
        device = self.model_manager.device
        threshold = self.model_manager.config.get("threshold", 0.4)

        if not self.nodule_groups:
            return {"type": "classification", "message": "未發現足夠的連續結節切片進行分類。"}

        results_summary = []
        all_nodule_probs = []
        self._reset_nodule_payload()
        first_group_rois = []   # for attention map — fixed to group 0
        first_group_fulls = []

        for g_idx, group in enumerate(self.nodule_groups):
            nodule_rois = []
            nodule_full_cts = []

            for idx in group:
                data = self.detection_results[idx]
                img = data["image"]
                if not data["boxes"]: continue

                # Use highest-confidence box (not just the first)
                best_box = max(data["boxes"], key=lambda b: b[4])
                x1, y1, x2, y2, _ = best_box
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # ROI: use actual detection size, clamped to [16, 64] half-side
                r = max(16, min(64, max((x2 - x1) // 2, (y2 - y1) // 2)))
                rx1 = max(0, cx - r);  ry1 = max(0, cy - r)
                rx2 = min(img.shape[1], cx + r); ry2 = min(img.shape[0], cy + r)
                roi = img[ry1:ry2, rx1:rx2]
                roi = cv2.resize(roi, (64, 64))

                # Context: 8× nodule radius crop → 128×128
                ctx_half = max(r * 8, 64)
                ctx_x1 = max(0, cx - ctx_half); ctx_x2 = min(img.shape[1], cx + ctx_half)
                ctx_y1 = max(0, cy - ctx_half); ctx_y2 = min(img.shape[0], cy + ctx_half)
                ctx_crop = img[ctx_y1:ctx_y2, ctx_x1:ctx_x2]
                full_ct = cv2.resize(ctx_crop if ctx_crop.size > 0 else img, (128, 128))

                # Preprocess: [0,255] → normalize (mean=0.5, std=0.5)
                roi_t  = (torch.from_numpy(roi).float().unsqueeze(0).unsqueeze(0)  / 255.0 - 0.5) / 0.5
                full_t = (torch.from_numpy(full_ct).float().unsqueeze(0).unsqueeze(0) / 255.0 - 0.5) / 0.5

                nodule_rois.append(roi_t.to(device))
                nodule_full_cts.append(full_t.to(device))

            # Save first group for attention map
            if g_idx == 0:
                first_group_rois  = nodule_rois
                first_group_fulls = nodule_full_cts

            # Multi-slice Gaussian-weighted integration
            n_slices = len(nodule_rois)
            if n_slices == 0: continue

            center = (n_slices - 1) / 2
            sigma  = max(n_slices / 6, 1.0)
            w = [np.exp(-((i - center) ** 2) / (2 * sigma ** 2)) for i in range(n_slices)]
            w = torch.tensor(w).to(device)
            w = w / w.sum()

            combined_probs = torch.zeros(2).to(device)
            with torch.no_grad():
                for i in range(n_slices):
                    outputs, _ = cnn_model(nodule_rois[i], nodule_full_cts[i])
                    combined_probs += F.softmax(outputs, dim=1)[0] * w[i]

            final_probs = combined_probs.cpu().numpy()
            mal_p = float(final_probs[1])
            lung_rads, label, action = _lung_rads(mal_p)

            results_summary.append(
                f"結節 #{g_idx+1}: {label}  [Lung-RADS {lung_rads}]  "
                f"惡性機率 {mal_p:.1%}  →  {action}"
            )
            all_nodule_probs.append(final_probs)
            # Structured payload for color-card rendering
            if not hasattr(self, "_last_nodules"):
                self._last_nodules = []
            self._last_nodules.append({
                "idx": g_idx + 1,
                "lung_rads": lung_rads,
                "label": label,
                "mal_prob": mal_p,
                "action": action,
                "n_slices": n_slices,
            })

        # Attention map: always from first (most prominent) group, center slice
        attention_map_bytes = None
        ct_overlay_bytes = None
        if first_group_rois and self.nodule_groups:
            mid = len(first_group_rois) // 2
            with torch.no_grad():
                _, att_maps = cnn_model(first_group_rois[mid], first_group_fulls[mid])
            attention_map_bytes = self._visualize_attention(first_group_rois[mid], att_maps)
            # Build full-CT overlay using middle slice of first nodule group
            try:
                first_group = self.nodule_groups[0]
                center_global_idx = first_group[len(first_group) // 2]
                slice_data = self.detection_results[center_global_idx]
                if slice_data["boxes"]:
                    best_box = max(slice_data["boxes"], key=lambda b: b[4])
                    ct_overlay_bytes = self._visualize_attention_on_ct(
                        slice_data["image"], best_box, att_maps)
            except Exception as e:
                print(f"CT attention overlay skipped: {e}")

        # Save JSON report to output_dir
        self._save_report(results_summary, all_nodule_probs,
                          self.model_manager.config.get("output_dir", "output"))

        # Generate summary message
        msg = "\n".join(results_summary)

        # Generate Chart
        chart_bytes = self._generate_chart(all_nodule_probs)

        return {
            "type": "classification",
            "message": f"分類完成！\n\n{msg}",
            "chart": chart_bytes,
            "attention_map": attention_map_bytes,
            "ct_with_attention": ct_overlay_bytes,
            "nodules": list(self._last_nodules),
        }

    def _draw_hud_boxes(self, result_data):  # noqa: E302
        """Draw professional medical HUD style boxes on the image."""
        img = result_data["image"]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        for (x1, y1, x2, y2, conf) in result_data["boxes"]:
            color = (2, 132, 199)  # #0284c7 in BGR
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 2)
            
            # Label background
            label = f"NODULE {conf:.1%}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_rgb, (x1, y1 - h - 10), (x1 + w + 10, y1), color, -1)
            cv2.putText(img_rgb, label, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        _, buffer = cv2.imencode(".png", img_rgb)
        return buffer.tobytes()

    def _visualize_attention_on_ct(self, ct_gray, yolo_box, att_maps):
        """Overlay CBAM spatial attention as a heatmap inside the YOLO bounding box,
        on the full CT slice image. Returns PNG bytes for the surgical viewport.
        """
        x1, y1, x2, y2, _conf = yolo_box
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(ct_gray.shape[1], int(x2)), min(ct_gray.shape[0], int(y2))
        if x2 <= x1 or y2 <= y1:
            _, buf = cv2.imencode(".png", cv2.cvtColor(ct_gray, cv2.COLOR_GRAY2BGR))
            return buf.tobytes()

        ct_bgr = cv2.cvtColor(ct_gray, cv2.COLOR_GRAY2BGR)
        bw, bh = x2 - x1, y2 - y1

        try:
            spatial_att = att_maps["roi_att2"][1]
            cam = spatial_att[0, 0].cpu().numpy()
            cam_min, cam_max = cam.min(), cam.max()
            if cam_max > cam_min:
                cam = (cam - cam_min) / (cam_max - cam_min)
            cam_u8 = (cam * 255).astype(np.uint8)
            cam_resized = cv2.resize(cam_u8, (bw, bh), interpolation=cv2.INTER_LINEAR)
            cam_color = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
            roi_region = ct_bgr[y1:y2, x1:x2]
            ct_bgr[y1:y2, x1:x2] = cv2.addWeighted(roi_region, 0.45, cam_color, 0.55, 0)
        except Exception as e:
            print(f"attention overlay fallback: {e}")

        # Draw the YOLO bounding box (white) and a corner accent
        cv2.rectangle(ct_bgr, (x1, y1), (x2, y2), (255, 255, 255), 2)
        for (cx, cy) in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
            cv2.line(ct_bgr, (cx - 6, cy), (cx + 6, cy), (255, 255, 255), 2)
            cv2.line(ct_bgr, (cx, cy - 6), (cx, cy + 6), (255, 255, 255), 2)

        _, buffer = cv2.imencode(".png", ct_bgr)
        return buffer.tobytes()

    def _visualize_attention(self, roi_tensor, att_maps=None):
        """Overlay real CBAM spatial attention (roi_att2) on the ROI image.

        att_maps: dict returned by NoduleClassifier.forward  — keys are
                  'roi_att1', 'roi_att2', ... each value is
                  (channel_att_map, spatial_att_map) from CBAM.
        Falls back to raw colormap when att_maps is empty or missing.
        """
        # De-normalize ROI: (x*0.5+0.5)*255
        roi_raw = (roi_tensor[0, 0].cpu().numpy() * 0.5 + 0.5)
        roi_raw = np.clip(roi_raw * 255, 0, 255).astype(np.uint8)
        roi_large = cv2.resize(roi_raw, (128, 128), interpolation=cv2.INTER_NEAREST)
        roi_bgr   = cv2.cvtColor(roi_large, cv2.COLOR_GRAY2BGR)

        try:
            # Use deepest ROI spatial attention map for best semantic signal
            spatial_att = att_maps["roi_att2"][1]        # (1, 1, H, W)
            cam = spatial_att[0, 0].cpu().numpy()        # (H, W)
            cam_min, cam_max = cam.min(), cam.max()
            if cam_max > cam_min:
                cam = (cam - cam_min) / (cam_max - cam_min)
            cam = (cam * 255).astype(np.uint8)
            cam = cv2.resize(cam, (128, 128), interpolation=cv2.INTER_LINEAR)
            cam_color = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(roi_bgr, 0.4, cam_color, 0.6, 0)
        except Exception:
            # Fallback when att_maps unavailable (e.g. model changed)
            overlay = cv2.applyColorMap(
                cv2.resize(roi_raw, (128, 128)), cv2.COLORMAP_JET)

        _, buffer = cv2.imencode(".png", overlay)
        return buffer.tobytes()

    def _save_report(self, results_summary, all_probs, output_dir):
        """Save a JSON report of classification results to output_dir."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            report = {
                "timestamp": timestamp,
                "model": "NoduleClassifier (CBAM)",
                "threshold": self.model_manager.config.get("threshold", 0.4),
                "nodules": []
            }
            for i, (summary, probs) in enumerate(zip(results_summary, all_probs)):
                lung_rads, label, action = _lung_rads(float(probs[1]))
                report["nodules"].append({
                    "id": i + 1,
                    "summary": summary,
                    "benign_prob": round(float(probs[0]), 4),
                    "malignant_prob": round(float(probs[1]), 4),
                    "lung_rads": lung_rads,
                    "prediction": label,
                    "recommended_action": action,
                })
            path = os.path.join(output_dir, f"report_{timestamp}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"報告儲存失敗: {e}")

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
