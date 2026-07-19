"""
MusicUnderstand 3.0 — Genre Head (音乐体裁分类)

输入 backbone 特征 [B, T, 256]，输出 10 类 GTZAN 体裁 logits。

结构:
  时间维平均池化 → MLP (256→128→num_genres)

GTZAN 体裁:
  blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
"""

import torch
import torch.nn as nn
from typing import List

# GTZAN 标准 10 个体裁
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']
NUM_GENRES = len(GENRES)  # 10

GENRE_TO_IDX = {g: i for i, g in enumerate(GENRES)}
IDX_TO_GENRE = {i: g for i, g in enumerate(GENRES)}


class GenreHead(nn.Module):
    """
    音乐体裁分类头

    Input:  [B, T, input_dim]  backbone 输出特征
    Output: [B, num_genres]    体裁 logits（CrossEntropyLoss 输入）
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        num_genres: int = NUM_GENRES,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_genres = num_genres

        # 时间维池化 + MLP 分类器
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # 单标签分类用 BN 效果好
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, num_genres),
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
            logits: [B, num_genres] 体裁 logits
        """
        # 时间维平均池化 [B, T, D] → [B, D]
        x = x.mean(dim=1)
        logits = self.net(x)  # [B, num_genres]
        return logits

    @torch.no_grad()
    def predict_genre(self, x: torch.Tensor) -> List[str]:
        """
        预测体裁名称（推理便捷接口）

        Args:
            x: [B, T, input_dim] backbone 特征

        Returns:
            genres: 体裁名称列表
        """
        logits = self.forward(x)
        pred_indices = logits.argmax(dim=-1).cpu().tolist()
        return [IDX_TO_GENRE[i] for i in pred_indices]


# ============================================================
# 单元测试
# ============================================================
if __name__ == "__main__":
    B, T = 4, 60
    dummy_feat = torch.randn(B, T, 256)

    head = GenreHead()
    logits = head(dummy_feat)
    probs = torch.softmax(logits, dim=-1)

    print(f"[OK] GenreHead 构建成功")
    print(f"  输入: {dummy_feat.shape}")
    print(f"  输出: {logits.shape}")
    print(f"  概率和: {probs.sum(dim=-1)}")  # 应接近 1.0

    # 测试推理接口
    genres = head.predict_genre(dummy_feat)
    print(f"  预测体裁: {genres}")

    total_params = sum(p.numel() for p in head.parameters())
    print(f"  参数: {total_params:,}")
