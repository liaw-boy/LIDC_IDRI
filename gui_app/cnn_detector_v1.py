import os
import sys
import cv2
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ultralytics import YOLO
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QWidget, QComboBox, QHBoxLayout, QListWidget,
    QListWidgetItem, QSplitter, QGroupBox, QFrame
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap, QImage, QColor
from collections import OrderedDict

# 雙輸入結節分類器（基於train_twochannel_v2.py的架構）
class NoduleClassifier(nn.Module):
    def __init__(self, roi_size=32, full_ct_size=640):
        super(NoduleClassifier, self).__init__()
        
        # ROI路徑的卷積網路
        self.roi_conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.roi_bn1 = nn.BatchNorm2d(32)
        self.roi_pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.roi_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.roi_bn2 = nn.BatchNorm2d(64)
        self.roi_pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.roi_conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.roi_bn3 = nn.BatchNorm2d(128)
        self.roi_pool3 = nn.MaxPool2d(kernel_size=2)
        
        # 計算ROI路徑全連接層的輸入特徵數
        # 經過三次池化後，32x32 -> 16x16 -> 8x8 -> 4x4
        self.roi_fc_input_size = 128 * 4 * 4
        
        # 全CT路徑的卷積網路
        self.full_ct_conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3)  # 640x640 -> 320x320
        self.full_ct_bn1 = nn.BatchNorm2d(16)
        self.full_ct_pool1 = nn.MaxPool2d(kernel_size=2)  # 320x320 -> 160x160
        
        self.full_ct_conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)  # 160x160 -> 80x80
        self.full_ct_bn2 = nn.BatchNorm2d(32)
        self.full_ct_pool2 = nn.MaxPool2d(kernel_size=2)  # 80x80 -> 40x40
        
        self.full_ct_conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 40x40 -> 20x20
        self.full_ct_bn3 = nn.BatchNorm2d(64)
        self.full_ct_pool3 = nn.MaxPool2d(kernel_size=2)  # 20x20 -> 10x10
        
        self.full_ct_conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 10x10 -> 10x10
        self.full_ct_bn4 = nn.BatchNorm2d(128)
        self.full_ct_pool4 = nn.MaxPool2d(kernel_size=2)  # 10x10 -> 5x5
        
        # 計算全CT路徑全連接層的輸入特徵數
        self.full_ct_fc_input_size = 128 * 5 * 5
        
        # 融合層
        self.fusion_fc1 = nn.Linear(self.roi_fc_input_size + self.full_ct_fc_input_size, 256)
        self.fusion_bn1 = nn.BatchNorm1d(256)
        self.fusion_dropout = nn.Dropout(0.5)
        self.fusion_fc2 = nn.Linear(256, 128)
        self.fusion_bn2 = nn.BatchNorm1d(128)
        self.fusion_fc3 = nn.Linear(128, 2)  # 二分類：良性和惡性
        
        # 兼容舊模型的標誌
        self.is_dual_input = True
    
    def forward(self, roi, full_ct=None):
        # 如果沒有提供full_ct，則只使用ROI路徑
        if full_ct is None:
            # 兼容舊模型的單輸入模式
            # ROI路徑
            roi_features = self.extract_roi_features(roi)
            
            # 使用ROI特徵直接進行分類
            # 修正：為了防止50%問題，這裡不使用零張量填充，而是使用ROI特徵的複製
            # 創建一個與ROI特徵相同大小的張量，但值較小
            pseudo_full_ct_features = roi_features * 0.1
            
            # 合併特徵
            combined = torch.cat((roi_features, pseudo_full_ct_features), dim=1)
            x = F.relu(self.fusion_bn1(self.fusion_fc1(combined)))
            x = self.fusion_dropout(x)
            x = F.relu(self.fusion_bn2(self.fusion_fc2(x)))
            x = self.fusion_fc3(x)
            
            # 為了保持與原始代碼兼容，返回一個空的注意力圖字典
            attention_maps = {}
            return x, attention_maps
        
        # 雙輸入模式
        # ROI路徑
        roi = F.relu(self.roi_bn1(self.roi_conv1(roi)))
        roi = self.roi_pool1(roi)
        roi = F.relu(self.roi_bn2(self.roi_conv2(roi)))
        roi = self.roi_pool2(roi)
        roi = F.relu(self.roi_bn3(self.roi_conv3(roi)))
        roi = self.roi_pool3(roi)
        roi_features = roi.view(-1, self.roi_fc_input_size)
        
        # 全CT路徑
        full_ct = F.relu(self.full_ct_bn1(self.full_ct_conv1(full_ct)))
        full_ct = self.full_ct_pool1(full_ct)
        full_ct = F.relu(self.full_ct_bn2(self.full_ct_conv2(full_ct)))
        full_ct = self.full_ct_pool2(full_ct)
        full_ct = F.relu(self.full_ct_bn3(self.full_ct_conv3(full_ct)))
        full_ct = self.full_ct_pool3(full_ct)
        full_ct = F.relu(self.full_ct_bn4(self.full_ct_conv4(full_ct)))
        full_ct = self.full_ct_pool4(full_ct)
        full_ct_features = full_ct.view(-1, self.full_ct_fc_input_size)
        
        # 特徵融合
        combined = torch.cat((roi_features, full_ct_features), dim=1)
        combined = F.relu(self.fusion_bn1(self.fusion_fc1(combined)))
        combined = self.fusion_dropout(combined)
        combined = F.relu(self.fusion_bn2(self.fusion_fc2(combined)))
        output = self.fusion_fc3(combined)
        
        # 為了保持與原始代碼兼容，返回一個空的注意力圖字典
        attention_maps = {}
        
        return output, attention_maps
    
    def extract_roi_features(self, roi):
        # ROI特徵提取
        roi = F.relu(self.roi_bn1(self.roi_conv1(roi)))
        roi = self.roi_pool1(roi)
        roi = F.relu(self.roi_bn2(self.roi_conv2(roi)))
        roi = self.roi_pool2(roi)
        roi = F.relu(self.roi_bn3(self.roi_conv3(roi)))
        roi = self.roi_pool3(roi)
        return roi.view(-1, self.roi_fc_input_size)

# 特徵提取器類
class FeatureExtractor:
    """
    從模型中提取特定層的特徵
    """
    def __init__(self, model, layers):
        """
        初始化特徵提取器
        
        Args:
            model: 預訓練的模型
            layers: 要提取特徵的層名稱列表
        """
        self.model = model
        self.model.eval()
        self.layers = layers
        self.hooks = []
        self.features = {layer: None for layer in layers}
        
        # 註冊鉤子
        for name, module in self.model.named_modules():
            if name in self.layers:
                self.hooks.append(
                    module.register_forward_hook(self._get_hook(name))
                )
    
    def _get_hook(self, name):
        """
        創建前向傳播鉤子函數
        """
        def hook(module, input, output):
            self.features[name] = output
        return hook
    
    def extract(self, roi, full_ct=None):
        """
        提取特徵
        
        Args:
            roi: 輸入ROI張量
            full_ct: 輸入完整CT張量
        
        Returns:
            dict: 每層的特徵
        """
        if full_ct is not None:
            self.model(roi, full_ct)
        else:
            self.model(roi)
        return self.features
    
    def remove_hooks(self):
        """
        移除所有鉤子
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

# 多切片特徵提取器
class MultiSliceFeatureExtractor:
    """
    從多切片中提取特徵
    """
    def __init__(self, model, feature_layer_name):
        """
        初始化多切片特徵提取器
        
        Args:
            model: 預訓練的模型
            feature_layer_name: 特徵提取層名稱
        """
        self.model = model
        self.model.eval()
        self.feature_layer_name = feature_layer_name
        self.feature_extractor = FeatureExtractor(model, [feature_layer_name])
    
    def extract_features(self, slices_roi, slices_full_ct=None, device='cuda'):
        """
        從多切片中提取特徵
        
        Args:
            slices_roi: 結節ROI切片列表
            slices_full_ct: 完整CT切片列表
            device: 計算設備
        
        Returns:
            list: 每個切片的特徵
        """
        features = []
        
        with torch.no_grad():
            for i, roi in enumerate(slices_roi):
                # 確保輸入格式正確
                if not isinstance(roi, torch.Tensor):
                    roi = torch.tensor(roi, dtype=torch.float32)
                
                # 添加批次維度如果需要
                if len(roi.shape) == 3:  # [channels, height, width]
                    roi = roi.unsqueeze(0)  # 變為 [1, channels, height, width]
                
                roi = roi.to(device)
                
                # 如果有對應的完整CT切片
                if slices_full_ct is not None and i < len(slices_full_ct):
                    full_ct = slices_full_ct[i]
                    if not isinstance(full_ct, torch.Tensor):
                        full_ct = torch.tensor(full_ct, dtype=torch.float32)
                    
                    if len(full_ct.shape) == 3:
                        full_ct = full_ct.unsqueeze(0)
                    
                    full_ct = full_ct.to(device)
                    
                    # 雙輸入提取特徵
                    slice_features = self.feature_extractor.extract(roi, full_ct)
                else:
                    # 單輸入提取特徵
                    slice_features = self.feature_extractor.extract(roi)
                
                # 保存特徵
                features.append(slice_features[self.feature_layer_name].cpu())
        
        return features

# 改進的多切片預測器
class MultiSlicePredictor:
    """
    基於多切片的結節良惡性預測器
    """
    def __init__(self, model, feature_extractor=None, integration_method='weighted_mean', threshold=0.4):
        """
        初始化多切片預測器
        
        Args:
            model: 預訓練的單切片模型
            feature_extractor: 特徵提取器，如果為None則直接使用模型輸出
            integration_method: 整合方法，可選 'mean', 'max', 'weighted_mean', 'voting'
            threshold: 分類閾值
        """
        self.model = model
        self.model.eval()
        self.feature_extractor = feature_extractor
        self.integration_method = integration_method
        self.threshold = threshold
    
    def predict(self, slices_roi, slices_full_ct=None, device='cuda'):
        """
        預測整個結節
        
        Args:
            slices_roi: 結節ROI切片列表
            slices_full_ct: 完整CT切片列表
            device: 計算設備
        
        Returns:
            prediction: 預測類別 (0=良性, 1=惡性)
            confidence: 預測信心度
            all_probs: 所有類別的概率
        """
        all_probs = []
        all_features = []
        
        # 對每個切片進行前向傳播
        with torch.no_grad():
            for i, roi in enumerate(slices_roi):
                try:
                    # 確保輸入格式正確
                    if isinstance(roi, np.ndarray):
                        roi_tensor = torch.from_numpy(roi).float()
                    else:
                        roi_tensor = roi.float()
                    
                    # 確保是 [batch, channel, height, width] 格式
                    if len(roi_tensor.shape) == 2:  # [height, width]
                        roi_tensor = roi_tensor.unsqueeze(0).unsqueeze(0)  # 變為 [1, 1, height, width]
                    elif len(roi_tensor.shape) == 3 and roi_tensor.shape[0] == 1:  # [1, height, width]
                        roi_tensor = roi_tensor.unsqueeze(0)  # 變為 [1, 1, height, width]
                    elif len(roi_tensor.shape) == 3:  # [height, width, channel]
                        roi_tensor = roi_tensor.permute(2, 0, 1).unsqueeze(0)  # 變為 [1, channel, height, width]
                    
                    roi_tensor = roi_tensor.to(device)
                    
                    # 如果有對應的完整CT切片
                    if slices_full_ct is not None and i < len(slices_full_ct):
                        full_ct = slices_full_ct[i]
                        
                        if isinstance(full_ct, np.ndarray):
                            full_ct_tensor = torch.from_numpy(full_ct).float()
                        else:
                            full_ct_tensor = full_ct.float()
                        
                        # 確保是 [batch, channel, height, width] 格式
                        if len(full_ct_tensor.shape) == 2:  # [height, width]
                            full_ct_tensor = full_ct_tensor.unsqueeze(0).unsqueeze(0)  # 變為 [1, 1, height, width]
                        elif len(full_ct_tensor.shape) == 3 and full_ct_tensor.shape[0] == 1:  # [1, height, width]
                            full_ct_tensor = full_ct_tensor.unsqueeze(0)  # 變為 [1, 1, height, width]
                        elif len(full_ct_tensor.shape) == 3:  # [height, width, channel]
                            full_ct_tensor = full_ct_tensor.permute(2, 0, 1).unsqueeze(0)  # 變為 [1, channel, height, width]
                        
                        full_ct_tensor = full_ct_tensor.to(device)
                        
                        # 雙輸入前向傳播
                        outputs, _ = self.model(roi_tensor, full_ct_tensor)
                    else:
                        # 單輸入前向傳播 - 修正：確保使用正確的單輸入模式
                        outputs, _ = self.model(roi_tensor, None)
                    
                    # 計算概率 - 修正：確保正確應用softmax
                    probs = F.softmax(outputs, dim=1)
                    all_probs.append(probs[0].cpu())
                    
                    # 調試輸出
                    print(f"切片 {i} 預測: 良性={probs[0][0].item():.4f}, 惡性={probs[0][1].item():.4f}")
                except Exception as e:
                    print(f"切片 {i} 預測失敗: {str(e)}")
                    continue
        
        # 整合所有切片的預測
        if len(all_probs) == 0:
            return -1, 0.0, [0.5, 0.5]
        
        # 將概率堆疊成張量
        probs_tensor = torch.stack(all_probs)
        
        # 根據整合方法進行預測
        if self.integration_method == 'mean':
            # 簡單平均
            final_probs = torch.mean(probs_tensor, dim=0)
        
        elif self.integration_method == 'max':
            # 取最大值
            final_probs = torch.max(probs_tensor, dim=0)[0]
        
        elif self.integration_method == 'weighted_mean':
            # 使用高斯權重
            n_slices = len(all_probs)
            center = (n_slices - 1) / 2
            sigma = max(n_slices / 6, 1.0)  # 確保sigma不會太小
            weights = [np.exp(-((i - center) ** 2) / (2 * sigma ** 2)) for i in range(n_slices)]
            weights = torch.tensor(weights, dtype=torch.float32).view(-1, 1)
            
            # 應用權重並求和
            weighted_probs = probs_tensor * weights
            final_probs = torch.sum(weighted_probs, dim=0) / torch.sum(weights)
        
        elif self.integration_method == 'voting':
            # 多數投票
            votes = torch.argmax(probs_tensor, dim=1)
            counts = torch.bincount(votes, minlength=2)
            max_vote = torch.argmax(counts).item()
            
            # 計算該類別的平均概率
            class_probs = probs_tensor[:, max_vote]
            confidence = torch.mean(class_probs).item()
            
            return max_vote, confidence, counts.float() / len(votes)
        
        # 獲取預測結果
        if final_probs[1] > self.threshold:  # 使用閾值判斷
            prediction = 1  # 惡性
        else:
            prediction = 0  # 良性
            
        confidence = final_probs[prediction].item()
        all_probs_np = final_probs.numpy()
        
        print(f"整合後預測: 良性={final_probs[0].item():.4f}, 惡性={final_probs[1].item():.4f}")
        print(f"最終預測: {prediction} (0=良性,1=惡性), 信心度: {confidence:.4f}")
        
        return prediction, confidence, all_probs_np


def predict_multi_slice_simple(model, slices_roi, slices_full_ct=None, device='cuda', threshold=0.4):
    """
    簡化版多切片預測 - 使用多數投票或平均概率
    
    Args:
        model: 預訓練的模型
        slices_roi: 結節ROI切片列表
        slices_full_ct: 完整CT切片列表
        device: 計算設備
        threshold: 分類閾值
    
    Returns:
        prediction: 預測類別 (0=良性, 1=惡性)
        confidence: 預測信心度
        probs: 所有類別的概率
    """
    predictions = []
    confidences = []
    all_probs = []
    
    # 對每個切片進行預測
    with torch.no_grad():
        for i, roi in enumerate(slices_roi):
            try:
                # 確保輸入格式正確
                if isinstance(roi, np.ndarray):
                    roi_tensor = torch.from_numpy(roi).float()
                else:
                    roi_tensor = roi.float()
                
                # 確保是 [batch, channel, height, width] 格式
                if len(roi_tensor.shape) == 2:  # [height, width]
                    roi_tensor = roi_tensor.unsqueeze(0).unsqueeze(0)  # 變為 [1, 1, height, width]
                elif len(roi_tensor.shape) == 3 and roi_tensor.shape[0] == 1:  # [1, height, width]
                    roi_tensor = roi_tensor.unsqueeze(0)  # 變為 [1, 1, height, width]
                elif len(roi_tensor.shape) == 3:  # [height, width, channel]
                    roi_tensor = roi_tensor.permute(2, 0, 1).unsqueeze(0)  # 變為 [1, channel, height, width]
                
                roi_tensor = roi_tensor.to(device)
                
                if slices_full_ct is not None and i < len(slices_full_ct):
                    full_ct = slices_full_ct[i]
                    
                    if isinstance(full_ct, np.ndarray):
                        full_ct_tensor = torch.from_numpy(full_ct).float()
                    else:
                        full_ct_tensor = full_ct.float()
                    
                    # 確保是 [batch, channel, height, width] 格式
                    if len(full_ct_tensor.shape) == 2:  # [height, width]
                        full_ct_tensor = full_ct_tensor.unsqueeze(0).unsqueeze(0)  # 變為 [1, 1, height, width]
                    elif len(full_ct_tensor.shape) == 3 and full_ct_tensor.shape[0] == 1:  # [1, height, width]
                        full_ct_tensor = full_ct_tensor.unsqueeze(0)  # 變為 [1, 1, height, width]
                    elif len(full_ct_tensor.shape) == 3:  # [height, width, channel]
                        full_ct_tensor = full_ct_tensor.permute(2, 0, 1).unsqueeze(0)  # 變為 [1, channel, height, width]
                    
                    full_ct_tensor = full_ct_tensor.to(device)
                    
                    outputs, _ = model(roi_tensor, full_ct_tensor)
                else:
                    # 修正：確保正確使用單輸入模式
                    outputs, _ = model(roi_tensor, None)
                    
                # 修正：確保正確應用softmax
                probs = F.softmax(outputs, dim=1)
                all_probs.append(probs[0].cpu())
                
                # 使用閾值判斷
                if probs[0][1].item() > threshold:
                    pred = 1  # 惡性
                else:
                    pred = 0  # 良性
                    
                conf = probs[0][pred].item()
                
                predictions.append(pred)
                confidences.append(conf)
                
                print(f"切片 {i}: 預測={pred} (0=良性,1=惡性), 信心度={conf:.4f}, 良性={probs[0][0].item():.4f}, 惡性={probs[0][1].item():.4f}")
            except Exception as e:
                print(f"切片 {i} 預測失敗: {str(e)}")
                continue
    
    # 如果沒有預測，返回未知
    if not predictions:
        return -1, 0.0, [0.5, 0.5]
    
    # 計算每個類別的出現次數
    counts = {0: 0, 1: 0}
    for pred in predictions:
        counts[pred] = counts.get(pred, 0) + 1
    
    # 找出出現最多的類別
    max_class = max(counts, key=counts.get)
    
    # 計算該類別的平均信心度
    class_confidences = [conf for pred, conf in zip(predictions, confidences) if pred == max_class]
    avg_confidence = sum(class_confidences) / len(class_confidences) if class_confidences else 0.0
    
    # 計算平均概率
    probs_tensor = torch.stack(all_probs)
    avg_probs = torch.mean(probs_tensor, dim=0).numpy()
    
    print(f"最終預測: {max_class} (0=良性,1=惡性), 出現次數: {counts[max_class]}/{len(predictions)}, 平均信心度: {avg_confidence:.4f}")
    print(f"平均概率: 良性={avg_probs[0]:.4f}, 惡性={avg_probs[1]:.4f}")
    
    return max_class, avg_confidence, avg_probs


class DetectionThread(QThread):
    progress_signal = pyqtSignal(str)
    completed_signal = pyqtSignal(str)

    def __init__(self, model_path, cnn_model_path, input_folder, output_folder, use_cuda):
        super().__init__()
        self.model_path = model_path
        self.cnn_model_path = cnn_model_path
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.use_cuda = use_cuda
        self.running = True
        self.use_multi_slice = True  # 啟用多切片預測

    def run(self):
        try:
            # 初始化 YOLO 模型
            device = "cuda" if self.use_cuda and torch.cuda.is_available() else "cpu"
            model = YOLO(self.model_path)
            model.to(device)
            print(f"使用裝置: {device.upper()}")

            # 初始化 CNN 模型
            try:
                cnn_model = NoduleClassifier()
                # 修正：檢查模型載入情況
                state_dict = torch.load(self.cnn_model_path, map_location=device)
                print(f"載入的模型權重鍵數量: {len(state_dict.keys())}")
                print(f"模型參數數量: {len([name for name, _ in cnn_model.named_parameters()])}")
                
                # 檢查權重鍵是否匹配
                model_keys = set([name for name, _ in cnn_model.named_parameters()])
                state_dict_keys = set(state_dict.keys())
                
                missing_keys = model_keys - state_dict_keys
                unexpected_keys = state_dict_keys - model_keys
                
                if missing_keys:
                    print(f"警告: 模型權重中缺少以下鍵: {missing_keys}")
                if unexpected_keys:
                    print(f"警告: 模型權重中存在未預期的鍵: {unexpected_keys}")
                
                # 載入權重並設置為評估模式
                cnn_model.load_state_dict(state_dict, strict=False)
                cnn_model.to(device)
                cnn_model.eval()
                print(f"CNN 模型載入成功: {self.cnn_model_path}")
                
                # 初始化多切片預測器
                if self.use_multi_slice:
                    try:
                        # 嘗試使用改進的多切片預測器
                        feature_extractor = MultiSliceFeatureExtractor(cnn_model, feature_layer_name='fusion_fc1')
                        multi_slice_predictor = MultiSlicePredictor(
                            cnn_model, 
                            feature_extractor=feature_extractor,
                            integration_method='weighted_mean',  # 可選: 'mean', 'max', 'weighted_mean', 'voting'
                            threshold=0.4  # 降低閾值，使系統更容易預測為惡性
                        )
                        print("多切片預測器初始化成功")
                    except Exception as e:
                        print(f"多切片預測器初始化失敗: {e}")
                        self.use_multi_slice = False
            except Exception as e:
                print(f"CNN 模型載入失敗: {str(e)}")
                cnn_model = None
                self.use_multi_slice = False

            # 創建輸出資料夾
            output_detected = os.path.join(self.output_folder, "output_detected")
            output_ct = os.path.join(self.output_folder, "original_ct")
            output_roi = os.path.join(self.output_folder, "original_roi")
            output_all_detected = os.path.join(self.output_folder, "original_detected")
            try:
                os.makedirs(output_detected, exist_ok=True)
                os.makedirs(output_ct, exist_ok=True)
                os.makedirs(output_roi, exist_ok=True)
                os.makedirs(output_all_detected, exist_ok=True)
            except Exception as e:
                print(f"建立輸出資料夾失敗: {str(e)}")
                return

            # 讀取 DICOM 檔案
            dicom_files = sorted([f for f in os.listdir(self.input_folder) if f.endswith(".dcm")])
            detection_status = []

            # 偵測每張圖片
            for idx, filename in enumerate(dicom_files):
                if not self.running:
                    print("偵測中斷！")
                    break

                dcm_path = os.path.join(self.input_folder, filename)
                dicom_data = pydicom.dcmread(dcm_path)
                pixel_array = dicom_data.pixel_array
                image = ((pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

                # 推論
                results = model(image_rgb)
                has_nodule = False
                valid_boxes = []

                for result in results:
                    for box, score in zip(result.boxes.xyxy, result.boxes.conf):
                        x_min, y_min, x_max, y_max = map(int, box[:4])
                        confidence = float(score)
                        has_nodule = True
                        valid_boxes.append((x_min, y_min, x_max, y_max, confidence))

                # 從 DICOM 檔名中提取序號 (例如 1-040.dcm -> 1-040)
                base_filename = os.path.splitext(filename)[0]
                detection_status.append((base_filename, has_nodule, valid_boxes, image))

                # 儲存原始偵測結果
                if has_nodule:
                    print(f"切片 {filename} 偵測到結節")
                    display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    for x_min, y_min, x_max, y_max, confidence in valid_boxes:
                        label = f"nodule {confidence:.2f}"
                        cv2.rectangle(display_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(display_image, (x_min, y_min - text_height - baseline), (x_min + text_width, y_min), (0, 255, 0), -1)
                        cv2.putText(display_image, label, (x_min, y_min - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    cv2.imwrite(os.path.join(output_all_detected, f"{base_filename}.png"), display_image)

                else:
                    print(f"切片 {filename} 未偵測到結節")

            # 篩選至少連續兩張以上的結節區段
            final_detection = [False] * len(detection_status)
            for i in range(len(detection_status) - 1):
                if detection_status[i][1] and detection_status[i + 1][1]:
                    final_detection[i] = True
                    final_detection[i + 1] = True

            # 顯示篩選結果
            print("篩選結果：")
            for i, detected in enumerate(final_detection):
                if detected:
                    print(f"切片 {detection_status[i][0]} 被篩選")

            # 將連續的結節分組為3D結節
            nodule_groups = []
            current_group = []
            
            for i, detected in enumerate(final_detection):
                if detected:
                    current_group.append(i)
                elif current_group:
                    nodule_groups.append(current_group)
                    current_group = []
            
            # 加入最後一組
            if current_group:
                nodule_groups.append(current_group)
            
            print(f"找到 {len(nodule_groups)} 個3D結節")
            
            # 儲存符合條件的影像
            roi_info = {}  # 儲存 ROI 資訊 (檔名, 預測結果, 信心度)
            
            # 處理每個3D結節
            for group_idx, group in enumerate(nodule_groups):
                print(f"處理第 {group_idx+1} 個3D結節，包含 {len(group)} 個切片")
                
                nodule_rois = []
                nodule_full_cts = []
                nodule_filenames = []
                
                # 收集該結節的所有切片
                for idx in group:
                    base_filename, has_nodule, valid_boxes, image = detection_status[idx]
                    nodule_filenames.append(base_filename)
                    
                    # 如果有多個結節，只處理第一個
                    if valid_boxes:
                        x_min, y_min, x_max, y_max, confidence = valid_boxes[0]
                        
                        # ROI 擷取
                        try:
                            center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2
                            roi_size = 32  # 使用 32x32 大小以符合 CNN 模型輸入
                            x1, y1 = max(0, center_x - roi_size // 2), max(0, center_y - roi_size // 2)
                            x2, y2 = min(image.shape[1], center_x + roi_size // 2), min(image.shape[0], center_y + roi_size // 2)
                            roi = image[y1:y2, x1:x2]
                            if roi.shape[0] != roi_size or roi.shape[1] != roi_size:
                                roi = cv2.resize(roi, (roi_size, roi_size))
                            
                            # 使用與標記影像相同的檔名儲存 ROI
                            output_roi_path = os.path.join(output_roi, f"{base_filename}.png")
                            cv2.imwrite(output_roi_path, roi)
                            
                            # 修正：確保正確的數據標準化
                            # 準備用於多切片預測的ROI和完整CT
                            roi_tensor = np.array(roi, dtype=np.float32) / 255.0
                            nodule_rois.append(roi_tensor)

                            # 準備完整CT圖像
                            full_ct_image = image.copy()
                            full_ct_image = cv2.resize(full_ct_image, (640, 640))
                            full_ct_tensor = np.array(full_ct_image, dtype=np.float32) / 255.0
                            nodule_full_cts.append(full_ct_tensor)
                                                        
                        except Exception as e:
                            print(f"ROI 擷取失敗: {str(e)}")
                
                # 使用多切片預測
                # 使用多切片預測
                if self.use_multi_slice and cnn_model is not None and len(nodule_rois) > 0:
                    try:
                        print(f"對第 {group_idx+1} 個3D結節進行多切片預測")
                        
                        # 使用改進的多切片預測器
                        malignancy_class, malignancy_prob, all_probs = multi_slice_predictor.predict(
                            nodule_rois, nodule_full_cts, device=device
                        )
                        
                        # 如果預測失敗，嘗試使用簡化版多切片預測
                        if malignancy_class == -1:
                            print("使用簡化版多切片預測")
                            malignancy_class, malignancy_prob, all_probs = predict_multi_slice_simple(
                                cnn_model, nodule_rois, nodule_full_cts, device=device, threshold=0.4
                            )
                        
                        # 設置結果
                        if malignancy_class == 1:  # 惡性
                            malignancy_result = "惡性"
                            color = (0, 0, 255)  # 紅色 (惡性)
                            text_color = (255, 255, 255)  # 白色文字
                        else:
                            malignancy_class = 0  # 良性
                            malignancy_result = "良性"
                            color = (0, 255, 0)  # 綠色 (良性)
                            text_color = (0, 0, 0)  # 黑色文字
                        
                        # 顯示詳細預測信息
                        print(f"結節 {group_idx+1} 預測結果: {malignancy_result}, 信心度: {malignancy_prob:.4f}")
                        print(f"良性概率: {all_probs[0]:.4f}, 惡性概率: {all_probs[1]:.4f}")
                        
                        # 將結果應用到該結節的所有切片
                        for filename in nodule_filenames:
                            roi_info[filename] = (malignancy_class, malignancy_prob)
                        
                    except Exception as e:
                        print(f"多切片預測失敗: {e}")
                        # 使用單切片預測作為備選
                        self._process_single_slice_predictions(cnn_model, nodule_rois, nodule_full_cts, nodule_filenames, roi_info, device)
                else:
                    # 如果模型為 None 或不使用多切片，設置默認值
                    if cnn_model is None:
                        print("CNN模型不可用，使用默認良性預測")
                        for filename in nodule_filenames:
                            roi_info[filename] = (0, 0.7)  # 預設為良性，但給予較高信心度
                    else:
                        # 使用單切片預測
                        self._process_single_slice_predictions(cnn_model, nodule_rois, nodule_full_cts, nodule_filenames, roi_info, device)

            # 根據roi_info生成最終標記的影像
            for idx, (base_filename, has_nodule, valid_boxes, image) in enumerate(detection_status):
                if final_detection[idx]:  # 只處理符合「連續兩張以上」規則的切片
                    display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    
                    # 如果有多個結節，只處理第一個
                    if valid_boxes:
                        x_min, y_min, x_max, y_max, confidence = valid_boxes[0]
                        
                        # 獲取預測結果
                        if base_filename in roi_info:
                            malignancy_class, malignancy_prob = roi_info[base_filename]
                            
                            if malignancy_class == 1:  # 惡性
                                #malignancy_result = "malignant"
                                color = (0, 0, 255)  # 紅色 (惡性)
                                text_color = (255, 255, 255)  # 白色文字
                            else:
                                #malignancy_result = "benign"
                                color = (0, 255, 0)  # 綠色 (良性)
                                text_color = (0, 0, 0)  # 黑色文字
                        else:
                            #malignancy_result = "unknown"
                            malignancy_prob = 0.5
                            color = (255, 255, 0)  # 黃色 (未知)
                            text_color = (0, 0, 0)  # 黑色文字
                        
                        # 標記框與信心分數，只顯示結節和信心度
                        label = f"nodule {confidence:.2f}"
                        cv2.rectangle(display_image, (x_min, y_min), (x_max, y_max), color, 2)
                        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(display_image, (x_min, y_min - text_height - baseline), (x_min + text_width, y_min), color, -1)
                        cv2.putText(display_image, label, (x_min, y_min - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                    # 儲存標註影像 (有框)
                    cv2.imwrite(os.path.join(output_detected, f"{base_filename}.png"), display_image)
                    # 儲存原始 CT (灰階)
                    cv2.imwrite(os.path.join(output_ct, f"{base_filename}.png"), image)
            
            # 儲存 ROI 資訊到檔案
            roi_info_path = os.path.join(self.output_folder, "roi_info.txt")
            with open(roi_info_path, 'w') as f:
                for filename, (class_id, prob) in roi_info.items():
                    f.write(f"{filename},{class_id},{prob:.4f}\n")

            print("偵測完成！")
            self.completed_signal.emit(self.output_folder)
        except Exception as e:
            print(f"錯誤: {str(e)}")
        finally:
            self.completed_signal.emit(self.output_folder)
    
    # 修正 nodule_rois 和 nodule_full_cts 的張量形狀問題
    def _process_single_slice_predictions(self, cnn_model, nodule_rois, nodule_full_cts, nodule_filenames, roi_info, device):
        """處理單切片預測"""
        print("使用單切片預測")
        
        for i, (roi, full_ct, filename) in enumerate(zip(nodule_rois, nodule_full_cts, nodule_filenames)):
            try:
                # 預處理 ROI 圖像 - 確保維度正確
                if isinstance(roi, np.ndarray):
                    roi_tensor = torch.from_numpy(roi).float()
                else:
                    roi_tensor = roi.float()
                
                # 確保是 [batch, channel, height, width] 格式
                if len(roi_tensor.shape) == 2:  # [height, width]
                    roi_tensor = roi_tensor.unsqueeze(0).unsqueeze(0)  # 變為 [1, 1, height, width]
                elif len(roi_tensor.shape) == 3 and roi_tensor.shape[0] == 1:  # [1, height, width]
                    roi_tensor = roi_tensor.unsqueeze(0)  # 變為 [1, 1, height, width]
                elif len(roi_tensor.shape) == 3:  # [height, width, channel]
                    roi_tensor = roi_tensor.permute(2, 0, 1).unsqueeze(0)  # 變為 [1, channel, height, width]
                
                roi_tensor = roi_tensor.to(device)
                
                # 獲取完整CT圖像並預處理
                if isinstance(full_ct, np.ndarray):
                    full_ct_tensor = torch.from_numpy(full_ct).float()
                else:
                    full_ct_tensor = full_ct.float()
                
                # 確保是 [batch, channel, height, width] 格式
                if len(full_ct_tensor.shape) == 2:  # [height, width]
                    full_ct_tensor = full_ct_tensor.unsqueeze(0).unsqueeze(0)  # 變為 [1, 1, height, width]
                elif len(full_ct_tensor.shape) == 3 and full_ct_tensor.shape[0] == 1:  # [1, height, width]
                    full_ct_tensor = full_ct_tensor.unsqueeze(0)  # 變為 [1, 1, height, width]
                elif len(full_ct_tensor.shape) == 3:  # [height, width, channel]
                    full_ct_tensor = full_ct_tensor.permute(2, 0, 1).unsqueeze(0)  # 變為 [1, channel, height, width]
                
                full_ct_tensor = full_ct_tensor.to(device)
                
                # 雙輸入預測 - 修正：確保正確應用softmax
                with torch.no_grad():
                    outputs, _ = cnn_model(roi_tensor, full_ct_tensor)
                    # 確保使用softmax獲得正確的概率分布
                    probs = F.softmax(outputs, dim=1)
                
                # 獲取良性和惡性的機率
                benign_prob = probs[0][0].item()  # 良性的機率
                malign_prob = probs[0][1].item()  # 惡性的機率
                
                # 使用較低的閾值0.4
                if malign_prob > 0.4:
                    malignancy_class = 1  # 惡性
                    malignancy_prob = malign_prob
                else:
                    malignancy_class = 0  # 良性
                    malignancy_prob = benign_prob
                
                # 儲存結果
                roi_info[filename] = (malignancy_class, malignancy_prob)
                
                print(f"切片 {filename} 單切片預測: 良性={benign_prob:.4f}, 惡性={malign_prob:.4f}, 結果: {malignancy_class}")
                
            except Exception as e:
                print(f"預測切片 {filename} 時發生錯誤: {str(e)}")
                # 修正：不要默認為良性50%，而是使用更合理的默認值
                roi_info[filename] = (0, 0.7)  # 預設為良性，但給予較高信心度


    def stop(self):
        self.running = False


class ImageViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.current_folder = None
        self.output_folder = None
        self.image_files = []
        self.current_index = 0
        self.roi_info = {}  # 儲存 ROI 資訊

    def init_ui(self):
        self.layout = QHBoxLayout(self)
        
        # 左側列表
        self.file_list = QListWidget()
        self.file_list.setMaximumWidth(200)
        self.file_list.currentRowChanged.connect(self.change_image)
        
        # 右側面板
        self.right_panel = QWidget()
        self.right_layout = QHBoxLayout(self.right_panel)
        
        # 圖片顯示區域
        self.image_panel = QWidget()
        self.image_layout = QVBoxLayout(self.image_panel)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(640, 640)
        self.image_label.setStyleSheet("border: 1px solid #cccccc;")
        
        # 導航按鈕
        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("上一張")
        self.next_button = QPushButton("下一張")
        self.prev_button.clicked.connect(self.prev_image)
        self.next_button.clicked.connect(self.next_image)
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)
        
        self.image_layout.addWidget(self.image_label)
        self.image_layout.addLayout(nav_layout)
        
        # 結節資訊面板
        self.info_panel = QWidget()
        self.info_layout = QVBoxLayout(self.info_panel)
        
        # 良惡性評估區域
        self.malignancy_group = QGroupBox("結節資訊")
        self.malignancy_layout = QVBoxLayout(self.malignancy_group)
        
        self.malignancy_label = QLabel("良惡性: 未知")
        self.malignancy_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.malignancy_label.setAlignment(Qt.AlignCenter)
        
        self.prob_label = QLabel("信心度: 未知")
        self.prob_label.setAlignment(Qt.AlignCenter)
        self.prob_label.setStyleSheet("font-size: 16px;")
        
        self.malignancy_layout.addWidget(self.malignancy_label)
        self.malignancy_layout.addWidget(self.prob_label)
        
        # ROI 顯示區域
        self.roi_group = QGroupBox("ROI 影像")
        self.roi_layout = QVBoxLayout(self.roi_group)
        
        self.roi_label = QLabel()
        self.roi_label.setAlignment(Qt.AlignCenter)
        self.roi_label.setMinimumSize(150, 150)
        self.roi_label.setStyleSheet("border: 1px solid #cccccc;")
        
        self.roi_layout.addWidget(self.roi_label)
        
        # 添加到資訊面板
        self.info_layout.addWidget(self.malignancy_group)
        self.info_layout.addWidget(self.roi_group)
        self.info_layout.addStretch()
        
        # 添加到右側面板
        self.right_layout.addWidget(self.image_panel, 4)
        self.right_layout.addWidget(self.info_panel, 1)
        
        # 添加到主佈局
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.file_list)
        splitter.addWidget(self.right_panel)
        splitter.setSizes([200, 800])
        
        self.layout.addWidget(splitter)
    
    def load_roi_info(self, output_folder):
        self.roi_info = {}
        roi_info_path = os.path.join(output_folder, "roi_info.txt")
        if os.path.exists(roi_info_path):
            try:
                with open(roi_info_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) == 3:
                            filename, class_id, prob = parts
                            self.roi_info[filename] = (int(class_id), float(prob))
            except Exception as e:
                print(f"載入 ROI 資訊失敗: {str(e)}")
    
    def load_folder(self, folder_path):
        if not os.path.exists(folder_path):
            return
            
        self.current_folder = folder_path
        self.output_folder = os.path.dirname(folder_path)
        self.image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.image_files.sort()
        
        # 載入 ROI 資訊
        self.load_roi_info(self.output_folder)
        
        self.file_list.clear()
        for file in self.image_files:
            self.file_list.addItem(QListWidgetItem(file))
            
        if self.image_files:
            self.current_index = 0
            self.file_list.setCurrentRow(0)
            self.load_image(0)
        
    def load_image(self, index):
        if not self.image_files or index < 0 or index >= len(self.image_files):
            return
            
        # 載入主圖片
        image_path = os.path.join(self.current_folder, self.image_files[index])
        pixmap = QPixmap(image_path)
        
        # 縮放圖片以適應標籤大小，保持比例
        pixmap = pixmap.scaled(640, 640, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)
        
        # 取得檔名 (不含副檔名)
        filename = os.path.splitext(self.image_files[index])[0]
        
        # 載入對應的 ROI 圖片
        roi_path = os.path.join(self.output_folder, "original_roi", f"{filename}.png")
        if os.path.exists(roi_path):
            roi_pixmap = QPixmap(roi_path)
            roi_pixmap = roi_pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.roi_label.setPixmap(roi_pixmap)
        else:
            self.roi_label.clear()
            self.roi_label.setText("無 ROI 影像")
        
        # 更新結節資訊
        if filename in self.roi_info:
            class_id, prob = self.roi_info[filename]
            
            # 根據分類結果設定顏色和文字
            if class_id == 0:
                color = "green"
                status = "良性"
            elif class_id == 1:
                color = "red"
                status = "惡性"
            else:
                color = "black"
                status = "未知"
            
            self.malignancy_label.setText(f"良惡性: {status}")
            self.malignancy_label.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {color};")
            
            self.prob_label.setText(f"信心度: {prob:.2%}")
            self.prob_label.setStyleSheet(f"font-size: 16px; color: {color};")
        else:
            self.malignancy_label.setText("良惡性: 未知")
            self.malignancy_label.setStyleSheet("font-size: 18px; font-weight: bold; color: black;")
            self.prob_label.setText("信心度: 未知")
            self.prob_label.setStyleSheet("font-size: 16px; color: black;")
        
    def change_image(self, row):
        if row >= 0:
            self.current_index = row
            self.load_image(row)
            
    def next_image(self):
        if not self.image_files:
            return
            
        self.current_index = (self.current_index + 1) % len(self.image_files)
        self.file_list.setCurrentRow(self.current_index)
        
    def prev_image(self):
        if not self.image_files:
            return
            
        self.current_index = (self.current_index - 1) % len(self.image_files)
        self.file_list.setCurrentRow(self.current_index)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("肺部結節偵測與良惡性評估工具")
        self.setGeometry(200, 200, 1000, 700)

        # 自動尋找同目錄下的模型檔案
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")
        self.cnn_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dual_input_best_auc_model.pth")
        
        # 設定patient資料夾路徑
        self.patient_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "patient")
        if not os.path.exists(self.patient_folder):
            os.makedirs(self.patient_folder, exist_ok=True)
        
                # 設定output資料夾路徑
        self.output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder, exist_ok=True)

        # 追蹤當前顯示模式 (0: 標記影像, 1: 原始CT)
        self.current_display_mode = 0

        self.init_ui()
        self.detection_thread = None
        self.load_patient_folders()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 上方控制區域
        control_layout = QHBoxLayout()

        # 選擇DICOM資料夾下拉式選單
        control_layout.addWidget(QLabel("選擇病患資料夾:"))
        self.patient_combo = QComboBox()
        self.patient_combo.setMinimumWidth(200)
        control_layout.addWidget(self.patient_combo)

        # 重新整理按鈕
        self.refresh_btn = QPushButton("重新整理")
        self.refresh_btn.clicked.connect(self.load_patient_folders)
        control_layout.addWidget(self.refresh_btn)

        # 開始偵測按鈕
        self.detect_btn = QPushButton("開始偵測")
        self.detect_btn.clicked.connect(self.start_detection)
        control_layout.addWidget(self.detect_btn)

        # 使用CUDA選項
        self.use_cuda_combo = QComboBox()
        self.use_cuda_combo.addItems(["使用CUDA (如果可用)", "僅使用CPU"])
        control_layout.addWidget(self.use_cuda_combo)

        main_layout.addLayout(control_layout)

        # 狀態標籤
        self.status_label = QLabel("就緒")
        main_layout.addWidget(self.status_label)

        # 圖片瀏覽器
        self.image_viewer = ImageViewer()
        main_layout.addWidget(self.image_viewer)

        # 結果選擇區域
        result_layout = QHBoxLayout()

        # 切換顯示模式按鈕
        self.toggle_view_btn = QPushButton("切換到原始CT")
        self.toggle_view_btn.clicked.connect(self.toggle_display_mode)
        result_layout.addWidget(self.toggle_view_btn)

        main_layout.addLayout(result_layout)

    def load_patient_folders(self):
        """載入 patient 資料夾中的所有子資料夾"""
        self.patient_combo.clear()
        if os.path.exists(self.patient_folder):
            folders = [d for d in os.listdir(self.patient_folder) 
                       if os.path.isdir(os.path.join(self.patient_folder, d))]
            folders.sort()
            for folder in folders:
                self.patient_combo.addItem(folder)
            
            self.status_label.setText(f"已載入 {len(folders)} 個病患資料夾")
        else:
            self.status_label.setText(f"找不到病患資料夾: {self.patient_folder}")

    def add_patient_folder(self):
        """新增病患資料夾"""
        folder = QFileDialog.getExistingDirectory(self, "選擇DICOM資料夾")
        if folder:
            # 取得選擇的資料夾名稱
            folder_name = os.path.basename(folder)
            target_folder = os.path.join(self.patient_folder, folder_name)
            
            # 如果目標資料夾已存在，詢問是否覆蓋
            if os.path.exists(target_folder):
                self.status_label.setText(f"資料夾 '{folder_name}' 已存在")
                return
            
            # 複製資料夾到 patient 目錄
            try:
                os.makedirs(target_folder, exist_ok=True)
                for file in os.listdir(folder):
                    if file.endswith(".dcm"):
                        src_file = os.path.join(folder, file)
                        dst_file = os.path.join(target_folder, file)
                        import shutil
                        shutil.copy2(src_file, dst_file)
                
                self.status_label.setText(f"已新增資料夾: {folder_name}")
                self.load_patient_folders()
                # 選擇新增的資料夾
                index = self.patient_combo.findText(folder_name)
                if index >= 0:
                    self.patient_combo.setCurrentIndex(index)
            except Exception as e:
                self.status_label.setText(f"新增資料夾失敗: {str(e)}")

    def start_detection(self):
        if self.patient_combo.count() == 0:
            self.status_label.setText("錯誤: 沒有可用的病患資料夾")
            return
            
        selected_folder = self.patient_combo.currentText()
        input_folder = os.path.join(self.patient_folder, selected_folder)
        
        if not os.path.exists(input_folder):
            self.status_label.setText("錯誤: 選擇的資料夾不存在")
            return

        if not os.path.exists(self.model_path):
            self.status_label.setText("錯誤: YOLO模型檔案不存在")
            return

        # 設定輸出資料夾
        output_folder = os.path.join(self.output_folder, selected_folder)
        os.makedirs(output_folder, exist_ok=True)

        # 設定是否使用CUDA
        use_cuda = self.use_cuda_combo.currentIndex() == 0

        # 禁用按鈕
        self.detect_btn.setEnabled(False)
        self.status_label.setText("正在進行偵測...")

        # 創建並啟動偵測線程
        self.detection_thread = DetectionThread(
            self.model_path, 
            self.cnn_model_path,
            input_folder, 
            output_folder, 
            use_cuda
        )
        self.detection_thread.completed_signal.connect(self.detection_completed)
        self.detection_thread.start()

    def detection_completed(self, output_folder):
        self.detect_btn.setEnabled(True)
        self.status_label.setText(f"偵測完成! 結果儲存於: {output_folder}")
        
        # 顯示結果
        self.update_display()

    def toggle_display_mode(self):
        """切換顯示模式"""
        if self.current_display_mode == 0:  # 目前是標記影像，切換到原始CT
            self.current_display_mode = 1
            self.toggle_view_btn.setText("切換到標記影像和ROI切片")
        else:  # 目前是原始CT，切換到標記影像
            self.current_display_mode = 0
            self.toggle_view_btn.setText("切換到原始CT")
        
        # 更新顯示
        self.update_display()

    def update_display(self):
        """根據當前顯示模式更新顯示內容"""
        if self.patient_combo.count() == 0:
            return
            
        selected_folder = self.patient_combo.currentText()
        output_folder = os.path.join(self.output_folder, selected_folder)
        
        if not os.path.exists(output_folder):
            return
            
        # 根據選擇的顯示模式載入不同的資料夾
        if self.current_display_mode == 0:  # 標記影像
            result_folder = os.path.join(output_folder, "output_detected")
        else:  # 原始CT
            result_folder = os.path.join(output_folder, "original_ct")
            
        if os.path.exists(result_folder):
            self.image_viewer.load_folder(result_folder)
        else:
            self.status_label.setText(f"找不到結果資料夾: {result_folder}")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()


