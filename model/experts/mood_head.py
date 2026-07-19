"""
MusicUnderstand — Mood/Theme Head (MTG-Jamendo 情绪/主题多标签)

输入 backbone 特征 [B, T, 256]，输出 MTG-Jamendo mood/theme 标签的多标签概率。

设计动机:
  DEAM 只有 1780 首歌，emotion_head 的 Valence 维度学不起来（R²≈0，语义信息
  不够）。MTG-Jamendo autotagging_moodtheme 子集有 ~18000 首歌、56个mood/theme
  标签（happy/sad/dark/uplifting/energetic/relaxing...），这些标签本身携带
  大量valence相关的语义信息。用这个多标签分类作辅助任务和 emotion_head 共享
  backbone 联合训练，让backbone学到的表征里包含更多语义/情绪线索，是给VA
  回归"输血"的手段——不是直接把标签转成VA坐标（那样引入的主观映射噪声太大，
  不如让模型自己在共享表征里学关联）。

结构: 与 GenreHead/InstrumentHead 一致（时间维平均池化 + MLP），
  输出 Sigmoid 多标签概率，训练用 BCEWithLogitsLoss。
"""

import torch
import torch.nn as nn
from typing import List


class MoodHead(nn.Module):
    """
    Mood/Theme 多标签检测头

    Input:  [B, T, input_dim]  backbone 输出特征
    Output: [B, num_tags]      未经 Sigmoid 的 logits (BCEWithLogitsLoss 输入)
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        num_tags: int = 56,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_tags = num_tags

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, num_tags),
        )

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
            x: [B, T, input_dim] backbone 时序特征

        Returns:
            logits: [B, num_tags]
        """
        x = x.mean(dim=1)
        return self.net(x)

    @torch.no_grad()
    def predict_tags(
        self, x: torch.Tensor, tag_names: List[str], threshold: float = 0.5
    ) -> List[List[str]]:
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        binary = (probs > threshold).cpu()

        predictions = []
        for i in range(binary.shape[0]):
            present = [tag_names[j] for j in range(self.num_tags) if binary[i, j]]
            predictions.append(present)
        return predictions


if __name__ == "__main__":
    B, T = 4, 30
    dummy_feat = torch.randn(B, T, 256)

    head = MoodHead(num_tags=56)
    logits = head(dummy_feat)
    print(f"[OK] MoodHead 构建成功")
    print(f"  输入: {dummy_feat.shape}")
    print(f"  输出: {logits.shape}")

    total_params = sum(p.numel() for p in head.parameters())
    print(f"  参数: {total_params:,}")
