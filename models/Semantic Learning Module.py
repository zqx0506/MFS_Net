import torch
import torch.nn as nn



class SelfEnhancingSemanticLearning(nn.Module):
    def __init__(self, in_channels, num_heads):
        super().__init__()
        # Self-attention + 动态卷积掩码模块
        self.self_attention = SelfAttentionModule(in_channels, num_heads)
        self.dynamic_mask_conv = DynamicMaskConv(in_channels)
        # Cross-modal attention + 动态掩码模块
        self.cross_attention = CrossAttentionModule(in_channels, num_heads)
        # Segmentation decoder (轻量MLP解码器)
        self.seg_decoder = LightweightSegDecoder(in_channels)

    def forward(self, x_self, x_cross):
        # 1. 单模态自增强：计算自注意力特征
        self_att_out = self.self_attention(x_self)  # Q,K,V 计算 + 注意力加权 + FFN

        # 2. 使用动态掩码调制卷积核，并卷积自注意力输出，强化局部结构
        mask_adjusted_out = self.dynamic_mask_conv(self_att_out)

        # 3. 分头操作，分别对每个头应用动态卷积，再融合多头特征
        multi_head_out = self.apply_multi_head_conv(self_att_out, mask_adjusted_out)

        # 4. 跨模态补充增强：用 x_cross 对多头融合特征进行 cross-attention + 动态掩码调制
        cross_att_out = self.cross_attention(x_cross, multi_head_out)

        # 5. 语义分割解码，输出语义类别预测
        seg_output = self.seg_decoder(cross_att_out)

        return seg_output

    def apply_multi_head_conv(self, attention_out, mask_out):
        # 伪代码：分头处理，卷积，拼接融合
        heads = split_heads(attention_out)
        conv_heads = [self.dynamic_mask_conv(head) for head in heads]
        fused = concat_heads(conv_heads) + mask_out
        return fused
