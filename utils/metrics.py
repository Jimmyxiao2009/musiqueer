"""
评价指标模块
用于评估情绪回归、体裁分类等任务
"""

import torch
import numpy as np
from typing import Tuple, Optional


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """R² 决定系数"""
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    return 1 - ss_res / (ss_tot + 1e-8)


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """均方根误差"""
    return torch.sqrt(((pred - target) ** 2).mean() + 1e-8)


def pearson_r(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Pearson 相关系数"""
    pred_centered = pred - pred.mean()
    target_centered = target - target.mean()
    cov = (pred_centered * target_centered).mean()
    std_p = pred_centered.std() + 1e-8
    std_t = target_centered.std() + 1e-8
    return cov / (std_p * std_t)


def concordance_ccc(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Concordance Correlation Coefficient (CCC)"""
    pred_mean = pred.mean()
    target_mean = target.mean()
    pred_var = pred.var()
    target_var = target.var()
    cov = ((pred - pred_mean) * (target - target_mean)).mean()
    return (2 * cov) / (pred_var + target_var + (pred_mean - target_mean) ** 2 + 1e-8)


def evaluate_emotion(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dim_names: Optional[list] = None,
) -> dict:
    """
    评估情绪回归各项指标

    Args:
        preds: [batch, T, 12] 预测值
        targets: [batch, T, 12] 真实值
        mask: [batch, T] 有效时间步掩码（可选）
        dim_names: 维度名称列表

    Returns:
        dict: 各维度 {指标名: 值}
    """
    if dim_names is None:
        dim_names = [
            "V_mean", "V_delta", "V_var",
            "A_mean", "A_delta", "A_var",
            "T_mean", "T_delta", "T_var",
            "C_mean", "C_delta", "C_var",
        ]

    if mask is not None:
        mask = mask.unsqueeze(-1)  # [B, T, 1]
        preds = preds * mask
        targets = targets * mask

    results = {}
    for d in range(preds.shape[-1]):
        p = preds[..., d].flatten()
        t = targets[..., d].flatten()
        # 跳过全零/无标注的维度
        if t.std() < 1e-6:
            continue
        results[f"{dim_names[d]}_R2"] = r2_score(p, t).item()
        results[f"{dim_names[d]}_RMSE"] = rmse(p, t).item()
        results[f"{dim_names[d]}_PearsonR"] = pearson_r(p, t).item()
        results[f"{dim_names[d]}_CCC"] = concordance_ccc(p, t).item()

    return results


def evaluate_genre(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 10,
) -> dict:
    """
    评估体裁分类指标

    Args:
        preds: [batch, num_classes] 概率输出
        targets: [batch] 类别索引
        num_classes: 类别数

    Returns:
        dict: accuracy, f1_macro
    """
    predicted = preds.argmax(dim=-1)
    correct = (predicted == targets).float().mean().item()

    # F1 macro 简化实现
    f1_scores = []
    for c in range(num_classes):
        tp = ((predicted == c) & (targets == c)).sum().float()
        fp = ((predicted == c) & (targets != c)).sum().float()
        fn = ((predicted != c) & (targets == c)).sum().float()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_scores.append(f1.item())

    f1_macro = np.mean(f1_scores)

    return {
        "accuracy": correct,
        "f1_macro": f1_macro,
    }
