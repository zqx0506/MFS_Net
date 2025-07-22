import torch
import torch.nn as nn



class FrequencyEnergyEnhancement(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 用于生成动态融合权重的轻量模块
        self.weight_predictor = LightweightConvBlock()
        # 双向跨频交互模块（如多头注意力）
        self.cross_attention = BidirectionalCrossAttention()

    def forward(self, ir_img, vis_img):
        # Step 1: 傅里叶变换，进入频域
        freq_ir = FFT(ir_img)
        freq_vis = FFT(vis_img)

        # Step 2: 高频/低频频带分解（基于掩码或滤波器）
        low_ir, high_ir = FrequencyDecompose(freq_ir)
        low_vis, high_vis = FrequencyDecompose(freq_vis)

        # Step 3: 交叉注意力增强不同模态频带间的互补信息
        enhanced_low_ir = self.cross_attention(low_ir, high_vis)
        enhanced_high_vis = self.cross_attention(high_vis, low_ir)

        # Step 4: 能量路由计算并生成像素级融合权重
        energy_diff = ComputeEnergyDifference(enhanced_low_ir, enhanced_high_vis)
        weights = self.weight_predictor(energy_diff)  # 输出权重图 W ∈ [0,1]

        # Step 5: 对增强结果进行频域融合，回到图像域
        fused_ir_part = IFFT(enhanced_low_ir)
        fused_vis_part = IFFT(enhanced_high_vis)
        fused = weights * fused_ir_part + (1 - weights) * fused_vis_part

        # Step 6: 与原始特征结合，分别用于 IR / VIS 分支
        out_vis = fused + vis_img
        out_ir = fused + ir_img

        return out_vis, out_ir, weights
