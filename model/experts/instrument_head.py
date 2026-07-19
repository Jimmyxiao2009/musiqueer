"""
MusicUnderstand 3.0 — Instrument Head (乐器检测)

输入 backbone 特征 [B, T, 256]，输出 10 类乐器多标签概率。

结构:
  时间维平均池化 → MLP (256→128→num_instruments)
  输出用 Sigmoid（多标签），训练用 BCEWithLogitsLoss

乐器:
  piano, drums, bass, guitar, violin, cello, flute, trumpet, saxophone, vocals
"""

import torch
import torch.nn as nn
from typing import List

# 10 种常见乐器
INSTRUMENTS = ['piano', 'drums', 'bass', 'guitar', 'violin',
               'cello', 'flute', 'trumpet', 'saxophone', 'vocals']
NUM_INSTRUMENTS = len(INSTRUMENTS)  # 10

INSTR_TO_IDX = {i: idx for idx, i in enumerate(INSTRUMENTS)}
IDX_TO_INSTR = {idx: i for idx, i in enumerate(INSTRUMENTS)}


class InstrumentHead(nn.Module):
    """
    乐器多标签检测头

    Input:  [B, T, input_dim]  backbone 输出特征
    Output: [B, num_instruments]  多标签概率（Sigmoid 输出）

    训练: BCEWithLogitsLoss
    推理: 输出已通过 Sigmoid，可设阈值（默认 0.5）得到二值标签
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        num_instruments: int = NUM_INSTRUMENTS,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_instruments = num_instruments

        # 时间维池化 + MLP 多标签分类器
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, num_instruments),
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
            logits: [B, num_instruments] 未经 Sigmoid 的 logits
                    训练时直接传入 BCEWithLogitsLoss
        """
        # 时间维平均池化 [B, T, D] → [B, D]
        x = x.mean(dim=1)
        logits = self.net(x)  # [B, num_instruments]
        return logits

    @torch.no_grad()
    def predict_instruments(
        self, x: torch.Tensor, threshold: float = 0.5
    ) -> List[List[str]]:
        """
        预测乐器标签（推理便捷接口）

        Args:
            x: [B, T, input_dim] backbone 特征
            threshold: Sigmoid 输出的二值化阈值

        Returns:
            predictions: 每样本存在的乐器名称列表
        """
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        binary = (probs > threshold).cpu()

        predictions = []
        for i in range(binary.shape[0]):
            present = [IDX_TO_INSTR[j] for j in range(self.num_instruments)
                       if binary[i, j]]
            predictions.append(present)
        return predictions


# ============================================================
# 单元测试
# ============================================================
if __name__ == "__main__":
    B, T = 4, 60
    dummy_feat = torch.randn(B, T, 256)

    head = InstrumentHead()
    logits = head(dummy_feat)
    probs = torch.sigmoid(logits)

    print(f"[OK] InstrumentHead 构建成功")
    print(f"  输入: {dummy_feat.shape}")
    print(f"  输出 (logits): {logits.shape}")
    print(f"  输出 (概率):   {probs.shape}")

    # 测试推理接口
    predictions = head.predict_instruments(dummy_feat)
    for i, pred in enumerate(predictions):
        print(f"  样本 {i}: {pred}")

    total_params = sum(p.numel() for p in head.parameters())
    print(f"  参数: {total_params:,}")
