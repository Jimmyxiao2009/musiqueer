"""
MusicUnderstand 2.0 — Emotion Head

从 backbone 特征提取 12 维情绪输出。

输出维度:
  [0]  Valence_mean
  [1]  Valence_delta     → 一阶差分（变化趋势）
  [2]  Valence_var       → 局部方差（波动幅度）
  [3]  Arousal_mean
  [4]  Arousal_delta
  [5]  Arousal_var
  [6]  Tension_mean
  [7]  Tension_delta
  [8]  Tension_var
  [9]  Complexity_mean
  [10] Complexity_delta
  [11] Complexity_var

Step 1 训练内容:
  - Valence_mean: DEAM 标注 → MSE
  - Arousal_mean: DEAM 标注 → MSE
  - Valence/Arousal delta/var: 从标注计算 ground truth → MSE
  - Tension: 暂不训练（无数据）
  - Complexity: 暂不训练（Phase 3 用伪标签）
"""

import torch
import torch.nn as nn


class EmotionHead(nn.Module):
    """
    12维情绪回归头

    Input:  [B, T, input_dim]  backbone输出的特征
    Output: [B, T, 12]         4维 × 3统计量（范围 0-1）
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        output_dim: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, output_dim),
        )

        # 注意：不用 Sigmoid，DEAM标注是Z-score归一化（范围约[-1, 1]）
        # Sigmoid会将输出限制在[0,1]，不匹配数据分布
        # 推理时可在外部做 [0,1] 映射
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, input_dim] backbone 输出

        Returns:
            emotion: [B, T, 12] 各维度在 [0, 1] 范围
        """
        emotion = self.net(x)  # [B, T, 12]
        return emotion


def compute_delta_var(seq: torch.Tensor) -> tuple:
    """
    从时序序列计算 delta（一阶差分）和 var（局部方差）

    Args:
        seq: [B, T] 时序序列（如 valence 或 arousal）

    Returns:
        delta: [B, T] 中心差分，边界用前向/后向补齐
        var:   [B, T] 5帧滑动窗口方差
    """
    B, T = seq.shape

    # --- Delta: 中心差分 ---
    delta = torch.zeros_like(seq)
    if T >= 3:
        # 内部点: 中心差分 (x[t+1] - x[t-1]) / 2
        delta[:, 1:-1] = (seq[:, 2:] - seq[:, :-2]) / 2.0
        # 左边界: 前向差分
        delta[:, 0] = seq[:, 1] - seq[:, 0]
        # 右边界: 后向差分
        delta[:, -1] = seq[:, -1] - seq[:, -2]

    # --- Var: 5帧滑动窗口方差 ---
    var = torch.zeros_like(seq)
    # 用 nn.AvgPool1d 近似计算局部方差
    # Var(x) = E[x²] - E[x]²
    x_sq = seq ** 2

    # 填充以适应边缘
    seq_pad = torch.nn.functional.pad(seq.unsqueeze(1), (2, 2), mode='replicate').squeeze(1)
    x_sq_pad = torch.nn.functional.pad(x_sq.unsqueeze(1), (2, 2), mode='replicate').squeeze(1)

    for t in range(T):
        window = seq_pad[:, t:t+5]      # [B, 5]
        window_sq = x_sq_pad[:, t:t+5]  # [B, 5]
        mean = window.mean(dim=1)
        mean_sq = window_sq.mean(dim=1)
        var[:, t] = mean_sq - mean ** 2

    var = torch.clamp(var, min=0.0)

    return delta, var
