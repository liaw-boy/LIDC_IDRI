# gui_app/nodule_classifier.py
"""NoduleClassifier implementations.

LegacyDualInputCNN  — deployed model: matches the trained .pth weights exactly (flat Conv).
NoduleClassifier    — upgraded CBAM Residual architecture (reference / future retraining).
AttentionModule / SpatialAttentionModule / ChannelAttentionModule / CBAM /
ResidualAttentionBlock — sub-modules used only by NoduleClassifier.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LegacyDualInputCNN(nn.Module):
    """Flat-Conv dual-input architecture that matches the trained .pth weights.

    ROI branch  (32×32)  → 2048-dim feature
    FullCT branch (640×640) → 3200-dim feature
    Fusion: concat(5248) → FC(256) → FC(128) → FC(2)
    """

    def __init__(self):
        super().__init__()
        # ROI branch
        self.roi_conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.roi_bn1   = nn.BatchNorm2d(32)
        self.roi_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.roi_bn2   = nn.BatchNorm2d(64)
        self.roi_conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.roi_bn3   = nn.BatchNorm2d(128)

        # Full-CT branch
        self.full_ct_conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3)
        self.full_ct_bn1   = nn.BatchNorm2d(16)
        self.full_ct_conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.full_ct_bn2   = nn.BatchNorm2d(32)
        self.full_ct_conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.full_ct_bn3   = nn.BatchNorm2d(64)
        self.full_ct_conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.full_ct_bn4   = nn.BatchNorm2d(128)

        # Fusion: 128*4*4 + 128*5*5 = 2048 + 3200 = 5248
        self.fusion_fc1 = nn.Linear(5248, 256)
        self.fusion_bn1 = nn.BatchNorm1d(256)
        self.fusion_fc2 = nn.Linear(256, 128)
        self.fusion_bn2 = nn.BatchNorm1d(128)
        self.fusion_fc3 = nn.Linear(128, 2)
        self.dropout    = nn.Dropout(0.5)

    def forward(self, roi, full_ct=None):
        # ROI: 32→16→8→4
        x = F.relu(self.roi_bn1(self.roi_conv1(roi)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.roi_bn2(self.roi_conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.roi_bn3(self.roi_conv3(x)))
        x = F.max_pool2d(x, 2)
        roi_feat = x.view(x.size(0), -1)  # 2048

        # Full CT: 640→320→160→80→40→20→5
        if full_ct is None:
            full_ct_feat = torch.zeros(roi.size(0), 3200, device=roi.device)
        else:
            y = F.relu(self.full_ct_bn1(self.full_ct_conv1(full_ct)))
            y = F.max_pool2d(y, 2)
            y = F.relu(self.full_ct_bn2(self.full_ct_conv2(y)))
            y = F.max_pool2d(y, 2)
            y = F.relu(self.full_ct_bn3(self.full_ct_conv3(y)))
            y = F.max_pool2d(y, 2)
            y = F.relu(self.full_ct_bn4(self.full_ct_conv4(y)))
            y = F.max_pool2d(y, 4)
            full_ct_feat = y.view(y.size(0), -1)  # 3200

        combined = torch.cat([roi_feat, full_ct_feat], dim=1)
        combined = F.relu(self.fusion_bn1(self.fusion_fc1(combined)))
        combined = self.dropout(combined)
        combined = F.relu(self.fusion_bn2(self.fusion_fc2(combined)))
        output = self.fusion_fc3(combined)
        return output, {}


# 注意力模組
class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        
    def forward(self, x):
        attention_map = torch.sigmoid(self.conv(x))
        return x * attention_map, attention_map


# 空間注意力模組
class SpatialAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels // 2)
        self.conv2 = nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels // 4)
        self.conv3 = nn.Conv2d(in_channels // 4, 1, kernel_size=1)
        
    def forward(self, x):
        feat = F.relu(self.bn1(self.conv1(x)))
        feat = F.relu(self.bn2(self.conv2(feat)))
        attention_map = torch.sigmoid(self.conv3(feat))
        return x * attention_map, attention_map


# 通道注意力模組
class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 確保reduction_ratio不會導致通道數為0
        reduced_channels = max(1, in_channels // reduction_ratio)
        
        self.fc1 = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        attention = self.sigmoid(out)
        return x * attention, attention


# 結合空間和通道注意力的CBAM模組
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttentionModule(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttentionModule(in_channels)
        
    def forward(self, x):
        x, channel_att_map = self.channel_att(x)
        x, spatial_att_map = self.spatial_att(x)
        return x, (channel_att_map, spatial_att_map)


# 殘差塊與注意力機制
class ResidualAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualAttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
        self.attention = CBAM(out_channels)
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out, att_maps = self.attention(out)
        out = F.relu(out)
        return out, att_maps


# 雙輸入結節分類器
class NoduleClassifier(nn.Module):
    def __init__(self, roi_size=32, full_ct_size=640,
                 use_attribute_feedback: bool = False, n_aux: int = 3,
                 in_channels: int = 1):
        super(NoduleClassifier, self).__init__()
        self.use_attribute_feedback = use_attribute_feedback
        self.n_aux = n_aux
        self.in_channels = in_channels

        # ROI路徑的卷積網路
        self.roi_conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.roi_bn1 = nn.BatchNorm2d(32)
        self.roi_pool1 = nn.MaxPool2d(kernel_size=2)
        
        # 第一個ROI注意力塊
        self.roi_att1 = ResidualAttentionBlock(32, 64, stride=1)
        self.roi_pool2 = nn.MaxPool2d(kernel_size=2)
        
        # 第二個ROI注意力塊
        self.roi_att2 = ResidualAttentionBlock(64, 128, stride=1)
        self.roi_pool3 = nn.MaxPool2d(kernel_size=2)

        # AdaptiveAvgPool 讓 ROI branch 輸出固定 4×4，不受輸入尺寸影響
        self.roi_adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # 計算ROI路徑全連接層的輸入特徵數（固定 4×4）
        self.roi_fc_input_size = 128 * 4 * 4
        
        # 全CT路徑的卷積網路（針對 128×128 context crop 設計）
        # 128 → pool(64) → ResAtt(64) → pool(32) → ResAtt(32) → pool(16) → ResAtt(16) → AdaptivePool(4×4)
        self.full_ct_conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.full_ct_bn1 = nn.BatchNorm2d(32)
        self.full_ct_pool1 = nn.MaxPool2d(kernel_size=2)  # 128 → 64

        # 第一個全CT注意力塊
        self.full_ct_att1 = ResidualAttentionBlock(32, 64, stride=1)   # 64 → 64
        self.full_ct_pool2 = nn.MaxPool2d(kernel_size=2)                # 64 → 32

        # 第二個全CT注意力塊
        self.full_ct_att2 = ResidualAttentionBlock(64, 128, stride=1)  # 32 → 32
        self.full_ct_pool3 = nn.MaxPool2d(kernel_size=2)                # 32 → 16

        # 第三個全CT注意力塊
        self.full_ct_att3 = ResidualAttentionBlock(128, 128, stride=1)  # 16 → 16
        # 移除 pool4：16 → AdaptivePool(4×4) 即可，保留更多空間資訊

        # AdaptiveAvgPool 讓 full_ct branch 輸出固定 4×4，不受輸入尺寸影響
        self.full_ct_adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # 計算全CT路徑全連接層的輸入特徵數
        self.full_ct_fc_input_size = 128 * 4 * 4  # 2048，固定大小
        
        # 融合層
        self.fusion_fc1 = nn.Linear(self.roi_fc_input_size + self.full_ct_fc_input_size, 256)
        self.fusion_bn1 = nn.BatchNorm1d(256)
        self.fusion_dropout = nn.Dropout(0.5)
        self.fusion_fc2 = nn.Linear(256, 128)
        self.fusion_bn2 = nn.BatchNorm1d(128)
        self.fusion_fc3 = nn.Linear(128, 2)  # 二分類：良性和惡性

        # Attribute Feedback heads (JIMI 2022) — optional, used when checkpoint provides them
        if self.use_attribute_feedback:
            self.aux_head = nn.Sequential(
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, n_aux),
                nn.Sigmoid(),
            )
            self.malignancy_head = nn.Linear(128 + n_aux, 2)

        # 兼容舊模型的標誌
        self.is_dual_input = True
    
    def forward(self, roi, full_ct=None):
        # 如果沒有提供full_ct，則只使用ROI路徑
        if full_ct is None:
            # 兼容模式：使用ROI提取特徵並做虛擬融合
            roi_features = self.extract_roi_features(roi)
            # 這裡簡單模擬融合行為以維持輸出結構一致
            pseudo_full_ct_features = torch.zeros((roi_features.shape[0], self.full_ct_fc_input_size)).to(roi.device)
            combined = torch.cat((roi_features, pseudo_full_ct_features), dim=1)
            x = F.relu(self.fusion_bn1(self.fusion_fc1(combined)))
            x = self.fusion_dropout(x)
            x = F.relu(self.fusion_bn2(self.fusion_fc2(x)))
            output = self.fusion_fc3(x)
            return output, {}
        
        # 雙輸入模式
        # ROI路徑
        roi = self.roi_pool1(F.relu(self.roi_bn1(self.roi_conv1(roi))))
        roi, roi_att1_maps = self.roi_att1(roi)
        roi = self.roi_pool2(roi)
        roi, roi_att2_maps = self.roi_att2(roi)
        roi = self.roi_pool3(roi)
        roi = self.roi_adaptive_pool(roi)  # → (B, 128, 4, 4) regardless of input size
        roi_features = roi.view(-1, self.roi_fc_input_size)
        
        # 全CT路徑
        full_ct = self.full_ct_pool1(F.relu(self.full_ct_bn1(self.full_ct_conv1(full_ct))))
        full_ct, full_ct_att1_maps = self.full_ct_att1(full_ct)
        full_ct = self.full_ct_pool2(full_ct)
        full_ct, full_ct_att2_maps = self.full_ct_att2(full_ct)
        full_ct = self.full_ct_pool3(full_ct)
        full_ct, full_ct_att3_maps = self.full_ct_att3(full_ct)
        full_ct = self.full_ct_adaptive_pool(full_ct)  # 16×16 → 4×4
        full_ct_features = full_ct.view(-1, self.full_ct_fc_input_size)
        
        # 特徵融合
        combined = torch.cat((roi_features, full_ct_features), dim=1)
        shared = F.relu(self.fusion_bn1(self.fusion_fc1(combined)))           # (B, 256)
        cls_feat = F.relu(self.fusion_bn2(self.fusion_fc2(self.fusion_dropout(shared))))  # (B, 128)

        if self.use_attribute_feedback:
            # JIMI 2022 Attribute Feedback path:
            # aux predictions feed back as explicit features into malignancy head
            aux_preds = self.aux_head(shared)                                  # (B, n_aux)
            output = self.malignancy_head(torch.cat([cls_feat, aux_preds], dim=1))
        else:
            output = self.fusion_fc3(cls_feat)

        # 收集所有注意力圖以供可視化
        attention_maps = {
            'roi_att1': roi_att1_maps,
            'roi_att2': roi_att2_maps,
            'full_ct_att1': full_ct_att1_maps,
            'full_ct_att2': full_ct_att2_maps,
            'full_ct_att3': full_ct_att3_maps
        }
        
        return output, attention_maps
    
    def extract_roi_features(self, roi):
        # ROI特徵提取
        roi = self.roi_pool1(F.relu(self.roi_bn1(self.roi_conv1(roi))))
        roi, _ = self.roi_att1(roi)
        roi = self.roi_pool2(roi)
        roi, _ = self.roi_att2(roi)
        roi = self.roi_pool3(roi)
        return roi.view(-1, self.roi_fc_input_size)
