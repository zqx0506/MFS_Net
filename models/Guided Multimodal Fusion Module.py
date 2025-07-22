import torch
import torch.nn as nn


class GuidedMultimodalFusion(nn.Module):
    def __init__(self):
        super().__init__()
        # 通道注意力和空间注意力模块
        self.channel_attention = ChannelAttentionBlock()
        self.spatial_attention = SpatialAttentionBlock()

    def forward(self, x1, x2):
        # Step 1: 生成通道注意力权重（基于全局池化+MLP）
        W_c = self.channel_attention(x1, x2)

        # Step 2: 生成空间注意力权重（基于最大/平均池化 + 卷积）
        W_s = self.spatial_attention(x1, x2)

        # Step 3: 融合注意力权重，得到混合门控图 g
        g = W_c * W_s  # 广播操作，得到最终门控图

        # Step 4: 提取两模态 Grad-CAM 风格响应图（用于显著图生成）
        M1 = ComputeSaliency(x1)  # 对应 x1 的响应热图
        M2 = ComputeSaliency(x2)  # 对应 x2 的响应热图

        # Step 5: 非线性融合（几何均值+加权和）
        M_fused = (0.5 * M1 + 0.5 * M2) * torch.sqrt(M1 * M2)

        # Step 6: 局部归一化（平滑+归一化）
        M_smooth = LocalNormalize(M_fused)  # 使用 3x3 平均池化 + MinMax 归一化

        # Step 7: 自适应阈值处理，生成二值掩码
        threshold = 0.5 * torch.mean(M_smooth)
        M_mask = (M_smooth >= threshold).float()

        # Step 8: Guided Fusion，根据掩码引导融合
        F_out = M_mask * x1 + (1 - M_mask) * x2

        # Step 9: 输出增强后的两模态特征
        x1_out = F_out * x1 + x1
        x2_out = F_out * x2 + x2

        return x1_out, x2_out, M_mask
