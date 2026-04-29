# gui_app/nodule_classifier.py
"""NoduleClassifier implementation with CBAM Attention and Residual Blocks.
Extracted from the final version in predict_3DCNN(1).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, roi_size=32, full_ct_size=640):
        super(NoduleClassifier, self).__init__()
        
        # ROI路徑的卷積網路
        self.roi_conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.roi_bn1 = nn.BatchNorm2d(32)
        self.roi_pool1 = nn.MaxPool2d(kernel_size=2)
        
        # 第一個ROI注意力塊
        self.roi_att1 = ResidualAttentionBlock(32, 64, stride=1)
        self.roi_pool2 = nn.MaxPool2d(kernel_size=2)
        
        # 第二個ROI注意力塊
        self.roi_att2 = ResidualAttentionBlock(64, 128, stride=1)
        self.roi_pool3 = nn.MaxPool2d(kernel_size=2)
        
        # 計算ROI路徑全連接層的輸入特徵數
        # 經過三次池化後，32x32 -> 16x16 -> 8x8 -> 4x4
        self.roi_fc_input_size = 128 * 4 * 4
        
        # 全CT路徑的卷積網路
        self.full_ct_conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3)  # 640x640 -> 320x320
        self.full_ct_bn1 = nn.BatchNorm2d(16)
        self.full_ct_pool1 = nn.MaxPool2d(kernel_size=2)  # 320x320 -> 160x160
        
        # 第一個全CT注意力塊
        self.full_ct_att1 = ResidualAttentionBlock(16, 32, stride=2)  # 160x160 -> 80x80
        self.full_ct_pool2 = nn.MaxPool2d(kernel_size=2)  # 80x80 -> 40x40
        
        # 第二個全CT注意力塊
        self.full_ct_att2 = ResidualAttentionBlock(32, 64, stride=2)  # 40x40 -> 20x20
        self.full_ct_pool3 = nn.MaxPool2d(kernel_size=2)  # 20x20 -> 10x10
        
        # 第三個全CT注意力塊
        self.full_ct_att3 = ResidualAttentionBlock(64, 128, stride=1)  # 10x10 -> 10x10
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
        roi_features = roi.view(-1, self.roi_fc_input_size)
        
        # 全CT路徑
        full_ct = self.full_ct_pool1(F.relu(self.full_ct_bn1(self.full_ct_conv1(full_ct))))
        full_ct, full_ct_att1_maps = self.full_ct_att1(full_ct)
        full_ct = self.full_ct_pool2(full_ct)
        full_ct, full_ct_att2_maps = self.full_ct_att2(full_ct)
        full_ct = self.full_ct_pool3(full_ct)
        full_ct, full_ct_att3_maps = self.full_ct_att3(full_ct)
        full_ct = self.full_ct_pool4(full_ct)
        full_ct_features = full_ct.view(-1, self.full_ct_fc_input_size)
        
        # 特徵融合
        combined = torch.cat((roi_features, full_ct_features), dim=1)
        combined = F.relu(self.fusion_bn1(self.fusion_fc1(combined)))
        combined = self.fusion_dropout(combined)
        combined = F.relu(self.fusion_bn2(self.fusion_fc2(combined)))
        output = self.fusion_fc3(combined)
        
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
