import torch
import torch.nn as nn


class GuidedMultimodalFusion(nn.Module):
    def __init__(self):
        super().__init__()

        self.channel_attention = ChannelAttentionBlock()
        self.spatial_attention = SpatialAttentionBlock()

    def forward(self, x1, x2):
        W_c = self.channel_attention(x1, x2)
        W_s = self.spatial_attention(x1, x2)
        g = W_c * W_s  
        M1 = ComputeSaliency(x1) 
        M2 = ComputeSaliency(x2) 
        M_fused = (0.5 * M1 + 0.5 * M2) * torch.sqrt(M1 * M2)
        M_smooth = LocalNormalize(M_fused)  
        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        F_out = ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        x1_out = F_out * x1 + x1
        x2_out = F_out * x2 + x2

        return x1_out, x2_out, M_mask
