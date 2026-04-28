import os
import gc
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc
)

# 1. 修改 EarlyStopping 類別 - 使用移動平均來平滑驗證損失
class EarlyStopping:
    """Early Stopping 機制，使用移動平均降低驗證損失波動"""
    def __init__(self, patience=20, verbose=True, delta=0.001, path='models/early_stop_checkpoint.pth', trace_func=print, smoothing_factor=0.6):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.smoothing_factor = smoothing_factor
        self.val_loss_history = []
        
        # 確保目錄存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
    
    def __call__(self, val_loss, model):
        # 將當前損失加入歷史記錄
        self.val_loss_history.append(val_loss)
        
        # 如果歷史記錄足夠長，計算移動平均
        if len(self.val_loss_history) >= 3:
            # 使用指數移動平均平滑驗證損失
            smoothed_loss = val_loss
            for i in range(1, min(5, len(self.val_loss_history))):
                weight = self.smoothing_factor ** i
                smoothed_loss = smoothed_loss * (1 - weight) + self.val_loss_history[-i-1] * weight
        else:
            smoothed_loss = val_loss
            
        score = -smoothed_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping 計數器: {self.counter} / {self.patience} (平滑後損失: {smoothed_loss:.6f})')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        """儲存模型當驗證損失減少時"""
        if self.verbose:
            self.trace_func(f'驗證損失減少 ({self.val_loss_min:.6f} --> {val_loss:.6f})。儲存模型...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# 2. 增強 EMA 實現 - 提高穩定性
class EMA:
    def __init__(self, model, decay=0.9998):  # 增加衰減係數以獲得更平滑的結果
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
                
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
                
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# 3. 改進的測試時增強 (TTA) - 更穩定的預測
def test_time_augmentation(model, roi_image, full_ct_image, device, num_augments=15):
    """增強版測試時增強：對同一張影像進行多次輕微增強並平均預測結果"""
    model.eval()
    
    # 基本轉換
    roi_tensor = torch.tensor(roi_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    full_ct_tensor = torch.tensor(full_ct_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # 標準化
    roi_tensor = transforms.Normalize((0.5,), (0.5,))(roi_tensor)
    full_ct_tensor = transforms.Normalize((0.5,), (0.5,))(full_ct_tensor)
    
    # 移動到設備
    roi_tensor = roi_tensor.to(device)
    full_ct_tensor = full_ct_tensor.to(device)
    
    # 初始預測
    all_probs = []
    
    with torch.no_grad():
        # 基本預測
        outputs = model(roi_tensor, full_ct_tensor)
        probs = F.softmax(outputs, dim=1)
        all_probs.append(probs.cpu().numpy())
        
        # 輕微增強預測 - 使用更多樣化但輕微的增強
        augmentations = [
            # 水平翻轉
            {'roi': transforms.RandomHorizontalFlip(p=1.0), 'full_ct': transforms.RandomHorizontalFlip(p=1.0)},
            # 垂直翻轉
            {'roi': transforms.RandomVerticalFlip(p=1.0), 'full_ct': transforms.RandomVerticalFlip(p=1.0)},
            # 輕微旋轉
            {'roi': transforms.RandomRotation(degrees=3), 'full_ct': transforms.RandomRotation(degrees=1)},
            # 輕微平移
            {'roi': transforms.RandomAffine(degrees=0, translate=(0.02, 0.02)), 
             'full_ct': transforms.RandomAffine(degrees=0, translate=(0.01, 0.01))},
            # 輕微縮放
            {'roi': transforms.RandomAffine(degrees=0, scale=(0.98, 1.02)), 
             'full_ct': transforms.RandomAffine(degrees=0, scale=(0.99, 1.01))},
            # 亮度調整
            {'roi': transforms.ColorJitter(brightness=0.05), 
             'full_ct': transforms.ColorJitter(brightness=0.03)},
            # 對比度調整
            {'roi': transforms.ColorJitter(contrast=0.05), 
             'full_ct': transforms.ColorJitter(contrast=0.03)},
        ]
        
        # 應用各種增強並預測
        for aug in augmentations:
            aug_roi = aug['roi'](roi_tensor)
            aug_full_ct = aug['full_ct'](full_ct_tensor)
            outputs = model(aug_roi, aug_full_ct)
            probs = F.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            
        # 額外的隨機組合增強
        for _ in range(num_augments - len(augmentations) - 1):
            # 隨機選擇並組合增強
            aug_roi = roi_tensor
            aug_full_ct = full_ct_tensor
            
            # 隨機應用一些增強
            if np.random.rand() > 0.5:
                aug_roi = transforms.RandomHorizontalFlip(p=0.5)(aug_roi)
                aug_full_ct = transforms.RandomHorizontalFlip(p=0.5)(aug_full_ct)
            
            if np.random.rand() > 0.7:
                aug_roi = transforms.RandomRotation(degrees=2)(aug_roi)
            
            if np.random.rand() > 0.7:
                aug_roi = transforms.RandomAffine(degrees=0, translate=(0.01, 0.01))(aug_roi)
                aug_full_ct = transforms.RandomAffine(degrees=0, translate=(0.005, 0.005))(aug_full_ct)
            
            outputs = model(aug_roi, aug_full_ct)
            probs = F.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
    
    # 平均預測結果 - 使用加權平均，基本預測權重較高
    weights = np.ones(len(all_probs))
    weights[0] = 2.0  # 原始影像預測權重加倍
    weighted_probs = np.average(all_probs, axis=0, weights=weights/np.sum(weights))
    
    return weighted_probs[0]

# 4. 改進的集成預測函數
def ensemble_prediction(model, roi_path, full_ct_path, roi_size=32, full_ct_size=640, device=None):
    """改進的集成預測，結合多個閾值和測試時增強"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 讀取並預處理ROI影像
    roi_image = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)
    roi_image = cv2.resize(roi_image, (roi_size, roi_size))
    roi_image = roi_image / 255.0  # 標準化到 0-1
    
    # 讀取並預處理全CT影像
    full_ct_image = cv2.imread(full_ct_path, cv2.IMREAD_GRAYSCALE)
    full_ct_image = cv2.resize(full_ct_image, (full_ct_size, full_ct_size))
    full_ct_image = full_ct_image / 255.0  # 標準化到 0-1
    
    # 使用增強版測試時增強獲取更穩定的預測
    probs = test_time_augmentation(model, roi_image, full_ct_image, device, num_augments=15)
    
    # 獲取惡性機率
    malignant_prob = probs[1]
    
    # 使用多個閾值進行集成決策 - 調整閾值以提高穩定性
    thresholds = {
        'high_confidence_benign': 0.25,  # 低於此值視為高置信度良性
        'standard': 0.593,               # 從ROC曲線獲得的最佳閾值
        'high_confidence_malignant': 0.75  # 高於此值視為高置信度惡性
    }
    
    # 根據機率和閾值進行決策
    if malignant_prob < thresholds['high_confidence_benign']:
        pred_class = 0  # 高置信度良性
        confidence = (1 - malignant_prob) * 100
        confidence_level = "高置信度"
    elif malignant_prob > thresholds['high_confidence_malignant']:
        pred_class = 1  # 高置信度惡性
        confidence = malignant_prob * 100
        confidence_level = "高置信度"
    else:
        # 中間區域，使用標準閾值
        pred_class = 1 if malignant_prob >= thresholds['standard'] else 0
        confidence = malignant_prob * 100 if pred_class == 1 else (1 - malignant_prob) * 100
        confidence_level = "中等置信度"
    
    # 轉換為類別名稱
    class_names = ['良性', '惡性']
    result = class_names[pred_class]
    
    return result, confidence, confidence_level, malignant_prob

# 5. 改進的評估函數 - 使用TTA提高穩定性
def evaluate_dual_input(model, dataloader, criterion, device):
    """改進的評估函數，使用測試時增強提高穩定性"""
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for roi_inputs, full_ct_inputs, labels in tqdm(dataloader, desc="評估"):
            roi_inputs, full_ct_inputs, labels = roi_inputs.to(device), full_ct_inputs.to(device), labels.to(device)
            
            batch_size = roi_inputs.size(0)
            batch_probs = []
            
            # 對每個樣本進行預測
            for i in range(batch_size):
                roi_input = roi_inputs[i:i+1]
                full_ct_input = full_ct_inputs[i:i+1]
                
                # 基本預測
                with autocast(device_type=device.type):
                    outputs = model(roi_input, full_ct_input)
                    probs = F.softmax(outputs, dim=1)
                
                # 對於驗證集，使用輕量級TTA (只做3次增強以節省時間)
                if batch_size <= 16:  # 假設小批次為驗證批次
                    # 水平翻轉
                    flipped_roi = torch.flip(roi_input, [3])
                    flipped_full_ct = torch.flip(full_ct_input, [3])
                    with autocast(device_type=device.type):
                        flip_outputs = model(flipped_roi, flipped_full_ct)
                        flip_probs = F.softmax(flip_outputs, dim=1)
                    
                    # 輕微旋轉 (使用預定義的旋轉矩陣以加速)
                    angle = 5.0
                    theta = torch.tensor([
                        [np.cos(np.radians(angle)), -np.sin(np.radians(angle)), 0],
                        [np.sin(np.radians(angle)), np.cos(np.radians(angle)), 0]
                    ], dtype=torch.float32).to(device)
                    
                    grid = F.affine_grid(theta.unsqueeze(0), roi_input.size(), align_corners=False)
                    rotated_roi = F.grid_sample(roi_input, grid, align_corners=False)
                    
                    with autocast(device_type=device.type):
                        rot_outputs = model(rotated_roi, full_ct_input)
                        rot_probs = F.softmax(rot_outputs, dim=1)
                    
                    # 平均所有預測
                    avg_probs = (probs + flip_probs + rot_probs) / 3.0
                else:
                    avg_probs = probs
                
                batch_probs.append(avg_probs.cpu().numpy())
            
            # 計算批次損失
            with autocast(device_type=device.type):
                outputs = model(roi_inputs, full_ct_inputs)
                loss = criterion(outputs, labels)
            
            # 統計
            running_loss += loss.item() * batch_size
            
            # 收集結果
            all_labels.extend(labels.cpu().numpy())
            
            # 使用平均預測結果
            batch_probs = np.vstack(batch_probs)
            batch_preds = np.argmax(batch_probs, axis=1)
            all_preds.extend(batch_preds)
            all_probs.extend(batch_probs[:, 1])  # 惡性類別的機率
    
    # 計算指標
    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc_score = roc_auc_score(all_labels, all_probs)
    
    return epoch_loss, accuracy, precision, recall, f1, auc_score, all_labels, all_preds, all_probs

# 6. 改進的訓練函數 - 使用循環學習率和漸進式學習
def train_dual_input_model(model, train_loader, test_loader, num_epochs=150, use_amp=True, patience=30):
    """改進的訓練函數，使用循環學習率和漸進式學習降低驗證波動"""
    # 檢查是否有 GPU 可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    # 將模型移到設備
    model = model.to(device)
    
    # 處理類別不平衡問題
    # 計算類別權重 - 使用更平衡的方式
    train_labels = [label for _, _, label in train_loader.dataset.data_list]
    class_counts = [train_labels.count(0), train_labels.count(1)]
    total_samples = sum(class_counts)
    # 使用平方根縮放來平衡類別權重
    class_weights = [1.0, np.sqrt(class_counts[0] / class_counts[1])] if class_counts[1] < class_counts[0] else [np.sqrt(class_counts[1] / class_counts[0]), 1.0]
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    print(f"類別權重: {class_weights.tolist()}")
    
    # 使用混合損失函數
    criterion = MixedLoss(alpha=0.6)  # 60% 焦點損失, 40% 標籤平滑交叉熵
    
    # 使用 AdamW 優化器，調整參數
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.0001,  # 更低的初始學習率
        weight_decay=5e-4,  # 增加權重衰減
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 使用循環學習率 - 有助於跳出局部最小值
    total_steps = len(train_loader) * num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=30 * len(train_loader),  # 每30個epoch重新啟動
        T_mult=1,  # 每次重啟後週期長度不變
        eta_min=1e-6  # 最小學習率
    )
    
    # 初始化 Early Stopping - 使用改進版
    early_stopping = EarlyStopping(patience=patience, verbose=True, delta=0.0005, 
                                  path='models/dual_input_early_stop_best_model.pth',
                                  smoothing_factor=0.7)  # 使用較高的平滑因子
    
    # 初始化 EMA - 使用更高的衰減係數
    ema = EMA(model, decay=0.9998)
    ema.register()
    
    # 初始化混合精度訓練
    if use_amp:
        scaler = GradScaler()
    else:
        scaler = None
    
    # 訓練模型
    best_val_auc = 0.0
    best_val_f1 = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # 建立儲存模型的資料夾
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # 梯度累積步數 - 增加批次大小的效果
    accumulation_steps = 4  # 減少累積步數以更頻繁更新
    
    # 實現漸進式學習 - 先用較大學習率快速收斂，再用小學習率微調
    training_phases = [
        {"epochs": 50, "description": "初始階段", "lr_scale": 1.0},
        {"epochs": 50, "description": "中間階段", "lr_scale": 0.5},
        {"epochs": 50, "description": "微調階段", "lr_scale": 0.1}
    ]
    
    current_epoch = 0
    for phase in training_phases:
        phase_epochs = phase["epochs"]
        lr_scale = phase["lr_scale"]
        
        # 調整學習率
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_scale
        
        print(f"\n開始 {phase['description']} (Epochs {current_epoch+1}-{current_epoch+phase_epochs})")
        print(f"學習率調整為: {optimizer.param_groups[0]['lr']:.6f}")
        
        for epoch in range(phase_epochs):
            overall_epoch = current_epoch + epoch
            print(f"\nEpoch {overall_epoch+1}/{num_epochs}")
            
            # 訓練一個 epoch
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            # 清除梯度
            optimizer.zero_grad()
            
            for step, (roi_inputs, full_ct_inputs, labels) in enumerate(tqdm(train_loader, desc="訓練")):
                roi_inputs, full_ct_inputs, labels = roi_inputs.to(device), full_ct_inputs.to(device), labels.to(device)
                
                if use_amp:
                    # 使用梯度累積
                    loss, outputs = train_with_gradient_accumulation(
                        model, roi_inputs, full_ct_inputs, labels, criterion, optimizer, scaler, accumulation_steps, step, len(train_loader), device
                    )
                else:
                    # 標準訓練
                    outputs = model(roi_inputs, full_ct_inputs)
                    loss = criterion(outputs, labels)
                    
                    # 反向傳播和優化
                    loss.backward()
                    
                    # 梯度裁剪，防止梯度爆炸
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    
                    if (step + 1) % accumulation_steps == 0 or (step + 1 == len(train_loader)):
                        optimizer.step()
                        optimizer.zero_grad()
                
                # 更新 EMA
                ema.update()
                
                # 更新學習率
                scheduler.step()
                
                # 統計
                running_loss += loss * roi_inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            # 計算訓練指標
            train_loss = running_loss / total
            train_acc = correct / total
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # 評估前應用 EMA 權重
            ema.apply_shadow()
            
            # 評估 - 使用改進的評估函數
            val_loss, val_acc, val_precision, val_recall, val_f1, val_auc, val_labels, val_preds, val_probs = evaluate_dual_input(
                model, test_loader, criterion, device
            )
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            # 恢復原始權重
            ema.restore()
            
            # 輸出當前學習率
            current_lr = optimizer.param_groups[0]['lr']
            print(f"當前學習率: {current_lr:.6f}")
            
            # 儲存最佳 AUC 模型
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                # 應用 EMA 權重再保存
                ema.apply_shadow()
                torch.save(model.state_dict(), 'models/dual_input_best_auc_model.pth')
                ema.restore()
                print(f"AUC 最佳模型已儲存! 最佳 AUC: {best_val_auc:.4f}")
            
            # 儲存最佳 F1 模型
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                # 應用 EMA 權重再保存
                ema.apply_shadow()
                torch.save(model.state_dict(), 'models/dual_input_best_f1_model.pth')
                ema.restore()
                print(f"F1 最佳模型已儲存! 最佳 F1: {best_val_f1:.4f}")
            
            # Early Stopping 檢查
            early_stopping(val_loss, model)
            
            # 輸出結果
            print(f"訓練損失: {train_loss:.4f}, 訓練準確率: {train_acc:.4f}")
            print(f"驗證損失: {val_loss:.4f}, 驗證準確率: {val_acc:.4f}, 精確度: {val_precision:.4f}, "
                  f"召回率: {val_recall:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
            
            # 輸出混淆矩陣
            cm = confusion_matrix(val_labels, val_preds)
            print(f"混淆矩陣:\n{cm}")
            
            # 如果 Early Stopping 觸發，則中斷訓練
            if early_stopping.early_stop:
                print(f"Early Stopping 觸發於 epoch {overall_epoch+1}")
                break
            
            # 釋放記憶體
            torch.cuda.empty_cache()
            gc.collect()
        
        # 如果 Early Stopping 已觸發，則中斷所有階段
        if early_stopping.early_stop:
            break
        
        # 更新當前 epoch 計數
        current_epoch += phase_epochs
    
    # 載入 Early Stopping 保存的最佳模型
    model.load_state_dict(torch.load('models/dual_input_early_stop_best_model.pth'))
    
    # 在測試集上進行最終評估 - 使用改進的評估函數
    test_loss, test_acc, test_precision, test_recall, test_f1, test_auc, test_labels, test_preds, test_probs = evaluate_dual_input(
        model, test_loader, criterion, device
    )
    
    print("\n最終測試結果 (Early Stopping 最佳模型):")
    print(f"損失: {test_loss:.4f}, 準確率: {test_acc:.4f}, 精確度: {test_precision:.4f}, "
          f"召回率: {test_recall:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}")
    
    # 也評估 AUC 最佳模型
    model.load_state_dict(torch.load('models/dual_input_best_auc_model.pth'))
    auc_test_loss, auc_test_acc, auc_test_precision, auc_test_recall, auc_test_f1, auc_test_auc, auc_test_labels, auc_test_preds, auc_test_probs = evaluate_dual_input(
        model, test_loader, criterion, device
    )
    
    print("\n最終測試結果 (AUC 最佳模型):")
    print(f"損失: {auc_test_loss:.4f}, 準確率: {auc_test_acc:.4f}, 精確度: {auc_test_precision:.4f}, "
          f"召回率: {auc_test_recall:.4f}, F1: {auc_test_f1:.4f}, AUC: {auc_test_auc:.4f}")
    
        # 評估 F1 最佳模型
    model.load_state_dict(torch.load('models/dual_input_best_f1_model.pth'))
    f1_test_loss, f1_test_acc, f1_test_precision, f1_test_recall, f1_test_f1, f1_test_auc, f1_test_labels, f1_test_preds, f1_test_probs = evaluate_dual_input(
        model, test_loader, criterion, device
    )
    
    print("\n最終測試結果 (F1 最佳模型):")
    print(f"損失: {f1_test_loss:.4f}, 準確率: {f1_test_acc:.4f}, 精確度: {f1_test_precision:.4f}, "
          f"召回率: {f1_test_recall:.4f}, F1: {f1_test_f1:.4f}, AUC: {f1_test_auc:.4f}")
    
    # 選擇最佳模型 - 綜合考量 AUC 和 F1
    best_models = {
        'early_stopping': {
            'path': 'models/dual_input_early_stop_best_model.pth',
            'auc': test_auc,
            'f1': test_f1,
            'combined_score': test_auc * 0.7 + test_f1 * 0.3  # 70% AUC + 30% F1
        },
        'best_auc': {
            'path': 'models/dual_input_best_auc_model.pth',
            'auc': auc_test_auc,
            'f1': auc_test_f1,
            'combined_score': auc_test_auc * 0.7 + auc_test_f1 * 0.3
        },
        'best_f1': {
            'path': 'models/dual_input_best_f1_model.pth',
            'auc': f1_test_auc,
            'f1': f1_test_f1,
            'combined_score': f1_test_auc * 0.7 + f1_test_f1 * 0.3
        }
    }
    
    # 找出綜合得分最高的模型
    best_model_key = max(best_models, key=lambda k: best_models[k]['combined_score'])
    best_model_path = best_models[best_model_key]['path']
    
    print(f"\n最佳整體模型: {best_model_key}")
    print(f"AUC: {best_models[best_model_key]['auc']:.4f}, F1: {best_models[best_model_key]['f1']:.4f}")
    print(f"綜合得分: {best_models[best_model_key]['combined_score']:.4f}")
    
    # 複製最佳模型到最終模型路徑
    shutil.copy(best_model_path, 'models/dual_input_final_model.pth')
    print(f"最佳模型已複製到: models/dual_input_final_model.pth")
    
    # 返回訓練歷史和最佳模型路徑
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_auc': best_val_auc,
        'best_val_f1': best_val_f1,
        'final_model_path': 'models/dual_input_final_model.pth'
    }
    
    return history, 'models/dual_input_final_model.pth'

# 7. 混合損失函數 - 結合多種損失函數提高穩定性
class MixedLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, smooth_factor=0.1):
        """
        混合損失函數：結合焦點損失和標籤平滑交叉熵
        
        參數:
            alpha: 焦點損失的權重
            gamma: 焦點損失的聚焦參數
            smooth_factor: 標籤平滑因子
        """
        super(MixedLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth_factor = smooth_factor
        self.ce = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, inputs, targets):
        # 標準交叉熵
        ce_loss = self.ce(inputs, targets)
        
        # 焦點損失部分
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # 標籤平滑部分
        num_classes = inputs.size(1)
        one_hot = torch.zeros_like(inputs)
        one_hot.scatter_(1, targets.unsqueeze(1), 1)
        
        # 應用標籤平滑
        smooth_one_hot = one_hot * (1 - self.smooth_factor) + self.smooth_factor / num_classes
        log_probs = F.log_softmax(inputs, dim=1)
        smooth_loss = -torch.sum(smooth_one_hot * log_probs, dim=1)
        
        # 混合損失
        mixed_loss = self.alpha * focal_loss + (1 - self.alpha) * smooth_loss
        
        return mixed_loss.mean()

# 8. 梯度累積訓練函數
def train_with_gradient_accumulation(model, roi_inputs, full_ct_inputs, labels, criterion, optimizer, scaler, accumulation_steps, current_step, total_steps, device):
    """使用梯度累積和混合精度訓練的單步訓練"""
    # 前向傳播
    with autocast(device_type=device.type):
        outputs = model(roi_inputs, full_ct_inputs)
        loss = criterion(outputs, labels) / accumulation_steps
    
    # 反向傳播
    scaler.scale(loss).backward()
    
    # 梯度累積
    if (current_step + 1) % accumulation_steps == 0 or (current_step + 1 == total_steps):
        # 梯度裁剪，防止梯度爆炸
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        # 更新參數
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    return loss * accumulation_steps, outputs

# 9. 改進的雙輸入模型 - 增加正則化和注意力機制
class DualInputCNN(nn.Module):
    def __init__(self, roi_size=32, full_ct_size=640, dropout_rate=0.4):
        super(DualInputCNN, self).__init__()
        
        # ROI 分支 - 使用更深的網絡
        self.roi_features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 全CT影像分支 - 使用更深的網絡
        self.full_ct_features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 注意力機制 - 自注意力層
        self.attention = SelfAttention(512)
        
        # 全連接層
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.75),  # 降低最後一層的 dropout
            nn.Linear(128, 2)
        )
        
        # 權重初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
    
    def forward(self, roi_input, full_ct_input):
        # 提取 ROI 特徵
        roi_features = self.roi_features(roi_input)
        roi_features = roi_features.view(roi_features.size(0), -1)
        
        # 提取全CT影像特徵
        full_ct_features = self.full_ct_features(full_ct_input)
        full_ct_features = full_ct_features.view(full_ct_features.size(0), -1)
        
        # 合併特徵
        combined_features = torch.cat((roi_features, full_ct_features), dim=1)
        
        # 應用注意力機制
        attended_features = self.attention(combined_features)
        
        # 分類
        output = self.classifier(attended_features)
        
        return output

# 10. 自注意力機制 - 增強特徵提取
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(in_dim, in_dim // 8)
        self.key = nn.Linear(in_dim, in_dim // 8)
        self.value = nn.Linear(in_dim, in_dim)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, dim = x.size()
        
        # 計算注意力分數
        proj_query = self.query(x).view(batch_size, -1, 1)  # B x C x 1
        proj_key = self.key(x).view(batch_size, -1, 1)  # B x C x 1
        energy = torch.bmm(proj_query, proj_key.transpose(1, 2))  # B x C x C
        attention = F.softmax(energy, dim=1)
        
        # 應用注意力
        proj_value = self.value(x).view(batch_size, -1, 1)  # B x C x 1
        out = torch.bmm(attention, proj_value).view(batch_size, -1)
        
        # 殘差連接
        out = self.gamma * out + x
        
        return out

# 11. 資料增強與預處理
def create_data_loaders(roi_dir, full_ct_dir, batch_size=32, roi_size=32, full_ct_size=640, val_split=0.2, test_split=0.1, seed=42):
    """改進的資料載入器，包含更強的資料增強"""
    # 設定隨機種子以確保可重複性
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 獲取所有檔案
    roi_files = sorted([f for f in os.listdir(roi_dir) if f.endswith('.png')])
    full_ct_files = sorted([f for f in os.listdir(full_ct_dir) if f.endswith('.png')])
    
    # 確保檔案數量相同
    assert len(roi_files) == len(full_ct_files), "ROI 和全CT影像檔案數量不匹配"
    
    # 從檔名中提取標籤 (假設檔名格式為 "patient_id_label.png")
    labels = []
    for f in roi_files:
        if '_benign' in f:
            labels.append(0)  # 良性
        elif '_malignant' in f:
            labels.append(1)  # 惡性
        else:
            raise ValueError(f"無法從檔名 {f} 中提取標籤")
    
    # 組合資料
    data = list(zip(roi_files, full_ct_files, labels))
    
    # 分層隨機分割資料
    train_data, temp_data = train_test_split(
        data, test_size=val_split+test_split, random_state=seed, stratify=labels
    )
    
    # 再次分割臨時資料以獲得驗證集和測試集
    val_ratio = val_split / (val_split + test_split)
    val_labels = [item[2] for item in temp_data]
    val_data, test_data = train_test_split(
        temp_data, test_size=1-val_ratio, random_state=seed, stratify=val_labels
    )
    
    # 定義訓練集增強 - 使用更強的增強
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # 定義 ROI 的訓練集增強 - 可以更激進
    roi_train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=20),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # 定義驗證/測試集轉換 - 只進行標準化
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # 創建資料集
    train_dataset = DualInputDataset(
        roi_dir, full_ct_dir, train_data, 
        roi_transform=roi_train_transform,
        full_ct_transform=train_transform,
        roi_size=roi_size, full_ct_size=full_ct_size
    )
    
    val_dataset = DualInputDataset(
        roi_dir, full_ct_dir, val_data,
        roi_transform=val_transform,
        full_ct_transform=val_transform,
        roi_size=roi_size, full_ct_size=full_ct_size
    )
    
    test_dataset = DualInputDataset(
        roi_dir, full_ct_dir, test_data,
        roi_transform=val_transform,
        full_ct_transform=val_transform,
        roi_size=roi_size, full_ct_size=full_ct_size
    )
    
    # 創建資料載入器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True, drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

# 12. 自定義資料集類別
class DualInputDataset(Dataset):
    def __init__(self, roi_dir, full_ct_dir, data_list, roi_transform=None, full_ct_transform=None, roi_size=32, full_ct_size=640):
        self.roi_dir = roi_dir
        self.full_ct_dir = full_ct_dir
        self.data_list = data_list
        self.roi_transform = roi_transform
        self.full_ct_transform = full_ct_transform
        self.roi_size = roi_size
        self.full_ct_size = full_ct_size
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        roi_file, full_ct_file, label = self.data_list[idx]
        
        # 讀取 ROI 影像
        roi_path = os.path.join(self.roi_dir, roi_file)
        roi_image = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)
        roi_image = cv2.resize(roi_image, (self.roi_size, self.roi_size))
        
        # 讀取全CT影像
        full_ct_path = os.path.join(self.full_ct_dir, full_ct_file)
        full_ct_image = cv2.imread(full_ct_path, cv2.IMREAD_GRAYSCALE)
        full_ct_image = cv2.resize(full_ct_image, (self.full_ct_size, self.full_ct_size))
        
        # 應用轉換
        if self.roi_transform:
            roi_image = self.roi_transform(roi_image)
        else:
            roi_image = torch.tensor(roi_image, dtype=torch.float32).unsqueeze(0) / 255.0
        
        if self.full_ct_transform:
            full_ct_image = self.full_ct_transform(full_ct_image)
        else:
            full_ct_image = torch.tensor(full_ct_image, dtype=torch.float32).unsqueeze(0) / 255.0
        
        return roi_image, full_ct_image, label

# 13. 主函數 - 整合所有功能
def main():
    """主函數 - 整合所有功能"""
    # 設定參數
    roi_dir = 'data/roi_images'
    full_ct_dir = 'data/full_ct_images'
    batch_size = 32
    roi_size = 32
    full_ct_size = 640
    num_epochs = 150
    
    # 確保結果目錄存在
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # 創建資料載入器
    train_loader, val_loader, test_loader = create_data_loaders(
        roi_dir, full_ct_dir, batch_size, roi_size, full_ct_size
    )
    
    # 創建模型
    model = DualInputCNN(roi_size, full_ct_size)
    
    # 訓練模型
    history, best_model_path = train_dual_input_model(
        model, train_loader, val_loader, num_epochs=num_epochs
    )
    
    # 載入最佳模型
    model.load_state_dict(torch.load(best_model_path))
    
    # 在測試集上評估
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = MixedLoss()
    
    test_loss, test_acc, test_precision, test_recall, test_f1, test_auc, test_labels, test_preds, test_probs = evaluate_dual_input(
        model, test_loader, criterion, device
    )
    
    print("\n最終測試結果:")
    print(f"損失: {test_loss:.4f}, 準確率: {test_acc:.4f}, 精確度: {test_precision:.4f}, "
          f"召回率: {test_recall:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}")
    
    # 繪製混淆矩陣
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['良性', '惡性'], yticklabels=['良性', '惡性'])
    plt.xlabel('預測')
    plt.ylabel('實際')
    plt.title('混淆矩陣')
    plt.savefig('results/confusion_matrix.png')
    
    # 繪製 ROC 曲線
    fpr, tpr, thresholds = roc_curve(test_labels, test_probs)
    roc_auc = auc(fpr, tpr)
    
    # 尋找最佳閾值 (約登指數最大點)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC 曲線 (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', 
               label=f'最佳閾值 = {optimal_threshold:.3f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假陽性率')
    plt.ylabel('真陽性率')
    plt.title('ROC 曲線')
    plt.legend(loc="lower right")
    plt.savefig('results/roc_curve.png')
    
    # 繪製訓練歷史
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='訓練')
    plt.plot(history['val_losses'], label='驗證')
    plt.title('損失')
    plt.xlabel('Epoch')
    plt.ylabel('損失')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_accs'], label='訓練')
    plt.plot(history['val_accs'], label='驗證')
    plt.title('準確率')
    plt.xlabel('Epoch')
    plt.ylabel('準確率')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/training_history.png')
    
    print("\n訓練完成! 結果已儲存到 results/ 資料夾")

if __name__ == "__main__":
    main()

