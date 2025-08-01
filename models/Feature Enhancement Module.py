import torch
import torch.nn as nn



class FrequencyEnergyEnhancement(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.weight_predictor = LightweightConvBlock()
        self.cross_attention = BidirectionalCrossAttention()

    def forward(self, ir_img, vis_img):
        freq_ir = FFT(ir_img)
        freq_vis = FFT(vis_img)

        low_ir, high_ir = FrequencyDecompose(freq_ir)
        low_vis, high_vis = FrequencyDecompose(freq_vis)

        enhanced_low_ir = self.cross_attention(low_ir, high_vis)
        enhanced_high_vis = self.cross_attention(high_vis, low_ir)

        energy_diff = ComputeEnergyDifference(enhanced_low_ir, enhanced_high_vis)
        weights = self.weight_predictor(energy_diff) 

        fused_ir_part = IFFT(enhanced_low_ir)
        fused_vis_part = IFFT(enhanced_high_vis)
        fused = weights * fused_ir_part + (1 - weights) * fused_vis_part
        out_vis = fused + vis_img
        out_ir = fused + ir_img

        return out_vis, out_ir, weights
