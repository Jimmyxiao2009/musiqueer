"""
MusicUnderstand 2.0 — Transformer Backbone

将 Mel 频谱编码为时序特征向量的主干网络。

架构设计:
  1. Patch Embedding (Conv2D 降维)
  2. 可学习位置编码
  3. Transformer Encoder × N 层 (Pre-LN结构)
  4. 输出: [batch, T, d_model]

Pre-LN（先 LayerNorm 再 Attention）比 Post-LN 更稳定，
训练时不需要 warmup，尤其适合小batch训练。
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class PatchEmbedding(nn.Module):
    """
    将 Mel 频谱切分为"patch"并映射到 d_model 维度

    Architecture:
      Mel [B, 1, 128, T]
        → Conv2D(1→64, k=3, s=1, p=1) + LayerNorm + GELU
        → Conv2D(64→128, k=(8,3), s=(8,1), p=(0,1))  # 频带压缩: 128→16
        → flatten freq+channel → [B, T, 2048]
        → Linear(2048, d_model) → [B, T, d_model]
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_mels: int = 128,
        conv1_channels: int = 64,
        conv2_channels: int = 128,
        freq_compression: int = 8,
        d_model: int = 256,
    ):
        super().__init__()

        # Conv2D 的频带压缩确保 n_mels 能被 freq_compression 整除
        assert n_mels % freq_compression == 0, (
            f"n_mels ({n_mels}) must be divisible by freq_compression ({freq_compression})"
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, conv1_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(conv1_channels),
            nn.GELU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(conv1_channels, conv2_channels,
                      kernel_size=(freq_compression, 3),
                      stride=(freq_compression, 1),
                      padding=(0, 1), bias=False),
            nn.BatchNorm2d(conv2_channels),
            nn.GELU(),
        )

        self.freq_dim = n_mels // freq_compression  # 16
        self.flatten_dim = conv2_channels * self.freq_dim  # 128*16 = 2048
        self.proj = nn.Linear(self.flatten_dim, d_model)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: [B, 1, n_mels, T] Mel 频谱

        Returns:
            x: [B, T, d_model] Patch 嵌入
        """
        x = self.conv1(mel)   # [B, 64, n_mels, T]
        x = self.conv2(x)     # [B, 128, n_mels/8, T] = [B, 128, 16, T]

        B, C_new, F, T = x.shape
        # Flatten channel×freq → [B, 128*16, T] → [B, T, 2048]
        x = x.reshape(B, C_new * F, T).permute(0, 2, 1)

        x = self.proj(x)  # [B, T, d_model]
        return x


class SinusoidalPositionalEncoding(nn.Module):
    """
    正弦位置编码（无需学习，可处理任意长度序列）
    相比可学习 Embedding，正弦编码对训练中未见过的长度更鲁棒。
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]

        Returns:
            x + positional encoding
        """
        return x + self.pe[:, :x.size(1), :]


class TransformerBlock(nn.Module):
    """
    Pre-LN Transformer Encoder Block

    Pre-LN 结构: LayerNorm → MHA → residual → LayerNorm → FFN → residual
    相比 Post-LN: 训练更稳定，梯度流更好，无需学习率 warmup
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
            src_mask: [T, T] or [B, T, T] 注意力掩码（可选）

        Returns:
            [B, T, d_model]
        """
        # Pre-LN Attention
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x),
                          attn_mask=src_mask, need_weights=False)[0]
        # Pre-LN FFN
        x = x + self.ffn(self.norm2(x))
        return x


class MusicTransformerBackbone(nn.Module):
    """
    MusicUnderstand 2.0 主干网络

    输入: [B, 1, n_mels, T] Mel频谱
    输出: [B, T, d_model] 音频表征
    
    结构:
        Mel → PatchEmbedding → PositionalEncoding → Transformer Encoder × N
    """

    def __init__(
        self,
        input_channels: int = 1,
        n_mels: int = 128,
        conv1_channels: int = 64,
        conv2_channels: int = 128,
        freq_compression: int = 8,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()

        self.d_model = d_model

        self.patch_embed = PatchEmbedding(
            in_channels=input_channels,
            n_mels=n_mels,
            conv1_channels=conv1_channels,
            conv2_channels=conv2_channels,
            freq_compression=freq_compression,
            d_model=d_model,
        )

        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_seq_len)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        """Xavier 初始化，有助于深层网络收敛"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        mel: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        return_all: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            mel: [B, 1, n_mels, T] Mel频谱
            src_mask: 注意力掩码（可选）
            return_all: 返回多尺度特征（用于 Decoder cross-attention）
                       False → [B, T, d_model] 仅最后一层
                       True  → [B, T, d_model*2] 浅层+深层拼接

        Multi-scale: 将第1层(局部细节)和第5-6层(全局语义)拼接
        让 Decoder 同时看到细粒度变化和整体结构。
        """
        x = self.patch_embed(mel)    # [B, T, d_model]
        x = self.pos_encoder(x)      # add positional encoding
        x = self.dropout(x)

        if not return_all:
            for block in self.blocks:
                x = block(x, src_mask)
            return x

        # Multi-scale: 保存中间层特征
        outputs = []
        for i, block in enumerate(self.blocks):
            x = block(x, src_mask)
            if i in (0, 4, 5):  # layer 1 (local), layer 5-6 (global)
                outputs.append(x)

        # 拼接多尺度特征: [B, T, d_model*3]
        # 后续会被 Decoder 的 Cross-Attention 使用
        multi_scale = torch.cat(outputs, dim=-1)
        # 投影回 d_model（保持维度一致）
        if not hasattr(self, 'fusion_proj'):
            self.fusion_proj = nn.Linear(self.d_model * 3, self.d_model).to(x.device)
        return self.fusion_proj(multi_scale)


# ============================================================
# 单元测试
# ============================================================
if __name__ == "__main__":
    B, T = 4, 60
    dummy_mel = torch.randn(B, 1, 128, T)

    model = MusicTransformerBackbone()
    output = model(dummy_mel)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[OK]  Transformer Backbone 构建成功")
    print(f"  输入:  {dummy_mel.shape}")
    print(f"  输出:  {output.shape}")
    print(f"  参数:  {total_params:,}")
    print(f"  每层:  {total_params // 6:,}")
