"""
MusicUnderstand 3.0 — 多任务联合训练脚本

Step 2-3: 同时训练 Emotion + Genre + Instrument

支持多数据集联合:
  - DEAM (emotion regression)
  - GTZAN (genre classification)
  - 可选 instrument 数据（当前用合成数据）

训练模式:
  - freeze_backbone: 加载预训练 backbone，只训练各 head
  - finetune: 加载预训练 backbone + emotion head，联合微调所有参数
  - end_to_end: 从头训练全部

损失函数:
  - Emotion: MSE (6 维 V/A mean/delta/var)
  - Genre: CrossEntropy
  - Instrument: BCEWithLogitsLoss (多标签)
  - 各任务 loss 可配置权重

用法:
  python train/train_multitask.py --epochs 100 --batch-size 32 \\
      --lr 0.001 --freeze-backbone --weight-emotion 1.0 --weight-genre 1.0
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Optional

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# 确保能找到项目模块
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

from model.music_understand import MusicUnderstand2
from data.deam_dataset import DEAMDataset, create_dataloaders as create_deam_loaders
from data.gtzan_dataset import GTZANDataset, create_gtzan_dataloaders
from model.experts.genre_head import GENRES, NUM_GENRES
from model.experts.instrument_head import INSTRUMENTS, NUM_INSTRUMENTS
from utils.metrics import evaluate_emotion, evaluate_genre

# --- 配置 ---
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "config", "default.yaml")
with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_emotion_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_weights: dict,
) -> torch.Tensor:
    """
    计算 12 维情绪损失的加权和

    Args:
        pred:   [B, T, 12] 模型预测
        target: [B, T, 12] 真实值
        mask:   [B, T]     有效帧掩码
        loss_weights: 各分量权重

    Returns:
        total_loss: 标量 emotion loss
        loss_dict:  各分量 loss 值
    """
    B, T, D = pred.shape
    mask_3d = mask.unsqueeze(-1).expand_as(pred)  # [B, T, 12]

    diff = (pred - target) ** 2
    diff = diff * mask_3d

    dim_losses = diff.sum(dim=(0, 1)) / (mask_3d.sum(dim=(0, 1)) + 1e-8)

    loss_dict = {
        "V_mean": dim_losses[0].item(),
        "V_delta": dim_losses[1].item(),
        "V_var": dim_losses[2].item(),
        "A_mean": dim_losses[3].item(),
        "A_delta": dim_losses[4].item(),
        "A_var": dim_losses[5].item(),
    }

    total_loss = (
        dim_losses[0] * loss_weights.get("valence_mean", 1.0) +
        dim_losses[3] * loss_weights.get("arousal_mean", 1.0) +
        dim_losses[1] * loss_weights.get("valence_delta", 0.5) +
        dim_losses[4] * loss_weights.get("arousal_delta", 0.5) +
        dim_losses[2] * loss_weights.get("valence_var", 0.3) +
        dim_losses[5] * loss_weights.get("arousal_var", 0.3)
    )

    return total_loss, loss_dict


def build_multitask_loss(
    emotion_pred: Optional[torch.Tensor],
    emotion_target: Optional[torch.Tensor],
    emotion_mask: Optional[torch.Tensor],
    genre_logits: Optional[torch.Tensor],
    genre_target: Optional[torch.Tensor],
    instr_logits: Optional[torch.Tensor],
    instr_target: Optional[torch.Tensor],
    loss_weights: Dict[str, float],
) -> Dict[str, torch.Tensor]:
    """
    计算多任务损失

    Args:
        emotion_pred:   [B, T, 12] 情绪预测（可选）
        emotion_target: [B, T, 12] 情绪目标（可选）
        emotion_mask:   [B, T] 有效掩码（可选）
        genre_logits:   [B, N_genre] 体裁 logits（可选）
        genre_target:   [B] 体裁标签（可选）
        instr_logits:   [B, N_instr] 乐器 logits（可选）
        instr_target:   [B, N_instr] 多标签目标（可选）
        loss_weights:   各任务权重 dict

    Returns:
        loss_dict: {
            "total": total_loss (加权和),
            "emotion": emotion_loss_value (已加权),
            "genre": genre_loss_value (已加权),
            "instrument": instrument_loss_value (已加权),
            "emotion_raw": raw emotion loss (仅统计),
            "genre_raw": raw genre loss,
            "instrument_raw": raw instrument loss,
        }
    """
    total_loss = torch.tensor(0.0, device=DEVICE)
    # 使用 dict 记录权重的 loss 值（用于反向传播）和 raw 值（用于日志）
    weighted_losses = {}
    raw_losses = {}

    # --- Emotion Loss (MSE) ---
    if emotion_pred is not None and emotion_target is not None:
        emo_w = loss_weights.get("emotion", 1.0)
        emo_loss, _ = build_emotion_loss(
            emotion_pred, emotion_target, emotion_mask,
            {k: v for k, v in loss_weights.items()
             if k in ("valence_mean", "arousal_mean",
                      "valence_delta", "arousal_delta",
                      "valence_var", "arousal_var")}
        )
        weighted_losses["emotion"] = emo_loss * emo_w
        raw_losses["emotion"] = emo_loss.detach()

    # --- Genre Loss (CrossEntropy) ---
    if genre_logits is not None and genre_target is not None:
        genre_w = loss_weights.get("genre", 1.0)
        genre_loss = nn.functional.cross_entropy(genre_logits, genre_target)
        weighted_losses["genre"] = genre_loss * genre_w
        raw_losses["genre"] = genre_loss.detach()

    # --- Instrument Loss (BCEWithLogits) ---
    if instr_logits is not None and instr_target is not None:
        instr_w = loss_weights.get("instrument", 1.0)
        instr_loss = nn.functional.binary_cross_entropy_with_logits(
            instr_logits, instr_target.float()
        )
        weighted_losses["instrument"] = instr_loss * instr_w
        raw_losses["instrument"] = instr_loss.detach()

    # 加总
    for v in weighted_losses.values():
        total_loss = total_loss + v

    loss_dict = {
        "total": total_loss,
        "emotion": weighted_losses.get("emotion", torch.tensor(0.0, device=DEVICE)),
        "genre": weighted_losses.get("genre", torch.tensor(0.0, device=DEVICE)),
        "instrument": weighted_losses.get("instrument", torch.tensor(0.0, device=DEVICE)),
        "emotion_raw": raw_losses.get("emotion", torch.tensor(0.0, device=DEVICE)),
        "genre_raw": raw_losses.get("genre", torch.tensor(0.0, device=DEVICE)),
        "instrument_raw": raw_losses.get("instrument", torch.tensor(0.0, device=DEVICE)),
    }

    return loss_dict


def train_epoch_multitask(
    model: nn.Module,
    deam_loader: torch.utils.data.DataLoader,
    gtzan_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    loss_weights: Dict[str, float],
    device: torch.device,
    grad_clip: float = 1.0,
    tasks: list = None,
) -> Dict[str, float]:
    """
    多任务训练一个 epoch

    使用交替批次训练策略:
      1. 交替从 DEAM 和 GTZAN 取 batch
      2. 每个 batch 只计算有数据的任务 loss
      3. 累积梯度后 step

    Args:
        tasks: 启用的任务列表 ['emotion', 'genre', 'instrument']

    Returns:
        avg_losses: 各任务平均 loss
    """
    if tasks is None:
        tasks = ['emotion', 'genre', 'instrument']

    model.train()
    total_losses = {
        "total": 0.0,
        "emotion": 0.0, "genre": 0.0, "instrument": 0.0,
        "emotion_raw": 0.0, "genre_raw": 0.0, "instrument_raw": 0.0,
    }

    # 创建两个 loader 的迭代器
    deam_iter = iter(deam_loader)
    gtzan_iter = iter(gtzan_loader)

    # 确定最大步数（以较长的 loader 为准）
    max_batches = max(len(deam_loader), len(gtzan_loader))
    n_batches = 0

    pbar = tqdm(range(max_batches), desc="Train", leave=False)
    for step in pbar:
        # --- DEAM Batch (emotion) ---
        try:
            deam_batch = next(deam_iter)
        except StopIteration:
            deam_iter = iter(deam_loader)
            deam_batch = next(deam_iter)

        mel_deam = deam_batch["mel"].to(device)          # [B, 1, 128, T]
        emotion_target = deam_batch["emotion"].to(device)  # [B, T, 12]
        emotion_mask = deam_batch["mask"].to(device)       # [B, T]

        active_experts = []
        if 'emotion' in tasks:
            active_experts.append('emotion')
        if 'genre' in tasks:
            active_experts.append('genre')
        if 'instrument' in tasks:
            active_experts.append('instrument')

        outputs = model(mel_deam, active_experts=active_experts)

        loss_dict = build_multitask_loss(
            emotion_pred=outputs.get('emotion'),
            emotion_target=emotion_target,
            emotion_mask=emotion_mask,
            genre_logits=outputs.get('genre_logits'),
            genre_target=None,      # DEAM 没有体裁标签
            instr_logits=outputs.get('instrument_logits'),
            instr_target=None,      # DEAM 没有乐器标签
            loss_weights=loss_weights,
        )

        # --- GTZAN Batch (genre) ---
        try:
            gtzan_batch = next(gtzan_iter)
        except StopIteration:
            gtzan_iter = iter(gtzan_loader)
            gtzan_batch = next(gtzan_iter)

        mel_gtzan = gtzan_batch["mel"].to(device)
        genre_target = gtzan_batch["genre"].to(device)

        outputs_g = model(mel_gtzan, active_experts=active_experts)

        loss_dict_g = build_multitask_loss(
            emotion_pred=outputs_g.get('emotion'),
            emotion_target=None,
            emotion_mask=None,
            genre_logits=outputs_g.get('genre_logits'),
            genre_target=genre_target,
            instr_logits=outputs_g.get('instrument_logits'),
            instr_target=None,      # GTZAN 没有乐器标签
            loss_weights=loss_weights,
        )

        # 合并两个 batch 的 loss
        total_loss = loss_dict["total"] + loss_dict_g["total"]

        optimizer.zero_grad()
        total_loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # 统计
        step_losses = {
            "total": total_loss.item(),
            "emotion": loss_dict["emotion"].item(),
            "genre": loss_dict_g["genre"].item(),
            "emotion_raw": loss_dict["emotion_raw"].item(),
            "genre_raw": loss_dict_g["genre_raw"].item(),
        }

        for k, v in step_losses.items():
            total_losses[k] += v

        n_batches += 1
        pbar.set_postfix({
            "total": f"{step_losses['total']:.4f}",
        })

    # 平均
    avg_losses = {k: v / n_batches for k, v in total_losses.items()}
    return avg_losses


@torch.no_grad()
def validate_multitask(
    model: nn.Module,
    deam_loader: torch.utils.data.DataLoader,
    gtzan_loader: torch.utils.data.DataLoader,
    loss_weights: Dict[str, float],
    device: torch.device,
    tasks: list = None,
) -> Dict[str, float]:
    """
    多任务验证

    Returns:
        metrics: 各任务指标 dict
    """
    if tasks is None:
        tasks = ['emotion', 'genre', 'instrument']

    model.eval()

    # 收集情绪预测用于详细评估
    all_emo_preds = []
    all_emo_targets = []
    all_emo_masks = []

    # 收集体裁预测用于详细评估
    all_genre_preds = []
    all_genre_targets = []

    total_losses = {
        "total": 0.0,
        "emotion": 0.0, "genre": 0.0, "instrument": 0.0,
        "emotion_raw": 0.0, "genre_raw": 0.0, "instrument_raw": 0.0,
    }
    n_batches_emo = 0
    n_batches_genre = 0

    # --- 验证情绪 ---
    if 'emotion' in tasks:
        for batch in tqdm(deam_loader, desc=" Val[Emo]", leave=False):
            mel = batch["mel"].to(device)
            emotion = batch["emotion"].to(device)
            mask = batch["mask"].to(device)

            outputs = model(mel, active_experts=['emotion'])
            pred = outputs['emotion']

            # loss
            loss_dict = build_multitask_loss(
                emotion_pred=pred, emotion_target=emotion,
                emotion_mask=mask,
                genre_logits=None, genre_target=None,
                instr_logits=None, instr_target=None,
                loss_weights=loss_weights,
            )

            total_losses["emotion"] += loss_dict["emotion"].item()
            total_losses["emotion_raw"] += loss_dict["emotion_raw"].item()
            n_batches_emo += 1

            all_emo_preds.append(pred.cpu())
            all_emo_targets.append(emotion.cpu())
            all_emo_masks.append(mask.cpu())

    # --- 验证体裁 ---
    if 'genre' in tasks:
        for batch in tqdm(gtzan_loader, desc=" Val[Gen]", leave=False):
            mel = batch["mel"].to(device)
            genre_target = batch["genre"].to(device)

            outputs = model(mel, active_experts=['genre'])
            genre_logits = outputs['genre_logits']

            loss_dict = build_multitask_loss(
                emotion_pred=None, emotion_target=None, emotion_mask=None,
                genre_logits=genre_logits, genre_target=genre_target,
                instr_logits=None, instr_target=None,
                loss_weights=loss_weights,
            )

            total_losses["genre"] += loss_dict["genre"].item()
            total_losses["genre_raw"] += loss_dict["genre_raw"].item()
            n_batches_genre += 1

            all_genre_preds.append(genre_logits.cpu())
            all_genre_targets.append(genre_target.cpu())

    # 平均 loss
    if n_batches_emo > 0:
        total_losses["emotion"] /= n_batches_emo
        total_losses["emotion_raw"] /= n_batches_emo
    if n_batches_genre > 0:
        total_losses["genre"] /= n_batches_genre
        total_losses["genre_raw"] /= n_batches_genre

    total_losses["total"] = total_losses["emotion"] + total_losses["genre"]

    # --- 计算详细指标 ---
    metrics = {}

    if len(all_emo_preds) > 0:
        preds_cat = torch.cat(all_emo_preds, dim=0)
        targets_cat = torch.cat(all_emo_targets, dim=0)
        masks_cat = torch.cat(all_emo_masks, dim=0)

        emotion_metrics = evaluate_emotion(
            preds_cat, targets_cat, masks_cat,
            dim_names=["V_mean", "V_delta", "V_var",
                        "A_mean", "A_delta", "A_var",
                        "T_mean", "T_delta", "T_var",
                        "C_mean", "C_delta", "C_var"],
        )
        metrics.update(emotion_metrics)

    if len(all_genre_preds) > 0:
        preds_cat = torch.cat(all_genre_preds, dim=0)
        targets_cat = torch.cat(all_genre_targets, dim=0)

        genre_metrics = evaluate_genre(preds_cat, targets_cat, NUM_GENRES)
        metrics.update(genre_metrics)

    return total_losses, metrics


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="MusicUnderstand 3.0 多任务训练"
    )
    parser.add_argument("--epochs", type=int, default=100,
                        help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="批次大小")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="学习率")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="冻结 backbone，只训练 head")
    parser.add_argument("--resume", type=str, default=None,
                        help="从 checkpoint 恢复")
    parser.add_argument("--weight-emotion", type=float, default=1.0,
                        help="Emotion 任务 loss 权重")
    parser.add_argument("--weight-genre", type=float, default=1.0,
                        help="Genre 任务 loss 权重")
    parser.add_argument("--weight-instrument", type=float, default=0.5,
                        help="Instrument 任务 loss 权重")
    parser.add_argument("--backbone-weights", type=str, default=None,
                        help="预训练 backbone 权重路径")
    parser.add_argument("--emotion-weights", type=str, default=None,
                        help="预训练 emotion head 权重路径")
    parser.add_argument("--gtzan-root", type=str, default="/root/data/gtzan",
                        help="GTZAN 数据根目录")
    parser.add_argument("--no-synthetic", action="store_true",
                        help="不使用合成数据（要求真实 GTZAN 数据存在）")
    parser.add_argument("--log-interval", type=int, default=5,
                        help="日志打印间隔（epoch）")
    parser.add_argument("--patience", type=int, default=15,
                        help="早停耐心值")
    return parser.parse_args()


def main():
    """主训练逻辑"""
    args = parse_args()

    print("=" * 60)
    print("   MusicUnderstand 3.0 — 多任务联合训练")
    print(f"   设备: {DEVICE}")
    print(f"   冻结 backbone: {args.freeze_backbone}")
    print(f"   任务权重: emo={args.weight_emotion}, "
          f"gen={args.weight_genre}, inst={args.weight_instrument}")
    print(f"   轮数: {args.epochs}, 批次: {args.batch_size}, lr: {args.lr}")
    print("=" * 60)

    # 损失权重
    loss_weights = {
        "emotion": args.weight_emotion,
        "genre": args.weight_genre,
        "instrument": args.weight_instrument,
        "valence_mean": 1.0,
        "arousal_mean": 1.0,
        "valence_delta": 0.5,
        "arousal_delta": 0.5,
        "valence_var": 0.3,
        "arousal_var": 0.3,
    }

    # 种子
    seed = CONFIG["train"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- 数据加载 ---
    print("\n[数据] 加载...")

    deam_root = CONFIG["paths"]["deam_root"]
    data_cfg = CONFIG["data"]
    train_cfg = CONFIG["train"]

    # 创建统一数据集（复用时共享相同的 mel 参数）
    deam_dataset_kwargs = {
        "root_dir": deam_root,
        "target_frames": data_cfg.get("target_frames", 30),
        "n_mels": data_cfg["n_mels"],
        "sample_rate": data_cfg["sample_rate"],
        "n_fft": data_cfg["n_fft"],
        "hop_length": data_cfg["hop_length"],
        "aggregation": data_cfg["frame_aggregation"],
        "max_duration": data_cfg["max_duration"],
    }

    train_deam = DEAMDataset(split="train", **deam_dataset_kwargs)
    val_deam = DEAMDataset(split="val", **deam_dataset_kwargs)

    gtzan_dataset_kwargs = {
        "root_dir": args.gtzan_root,
        "target_frames": data_cfg.get("target_frames", 30),
        "n_mels": data_cfg["n_mels"],
        "sample_rate": data_cfg["sample_rate"],
        "n_fft": data_cfg["n_fft"],
        "hop_length": data_cfg["hop_length"],
        "aggregation": data_cfg["frame_aggregation"],
        "max_duration": data_cfg["max_duration"],
    }

    train_gtzan = GTZANDataset(
        split="train",
        synthetic_on_fail=not args.no_synthetic,
        **gtzan_dataset_kwargs,
    )
    val_gtzan = GTZANDataset(
        split="val",
        synthetic_on_fail=not args.no_synthetic,
        **gtzan_dataset_kwargs,
    )

    print(f"  DEAM 训练: {len(train_deam)} 段, 验证: {len(val_deam)} 段")
    print(f"  GTZAN 训练: {len(train_gtzan)} 首, 验证: {len(val_gtzan)} 首")

    train_emo_loader = torch.utils.data.DataLoader(
        train_deam, batch_size=args.batch_size,
        shuffle=True, num_workers=train_cfg.get("num_workers", 4),
    )
    val_emo_loader = torch.utils.data.DataLoader(
        val_deam, batch_size=args.batch_size,
        shuffle=False, num_workers=train_cfg.get("num_workers", 4),
    )
    train_genre_loader = torch.utils.data.DataLoader(
        train_gtzan, batch_size=args.batch_size,
        shuffle=True, num_workers=train_cfg.get("num_workers", 4),
    )
    val_genre_loader = torch.utils.data.DataLoader(
        val_gtzan, batch_size=args.batch_size,
        shuffle=False, num_workers=train_cfg.get("num_workers", 4),
    )

    # --- 模型 ---
    print("\n[模型] 初始化...")

    backbone_cfg = {k: v for k, v in CONFIG["model"]["backbone"].items()
                    if k != "type"}
    emotion_cfg = {k: v for k, v in CONFIG["model"]["experts"]["emotion"].items()
                   if k not in ("loss_weight",)}

    model = MusicUnderstand2(
        backbone_config=backbone_cfg,
        emotion_config=emotion_cfg,
        genre_config={},         # 使用默认参数
        instrument_config={},    # 使用默认参数
    ).to(DEVICE)

    # 加载预训练权重
    if args.backbone_weights:
        ckpt = torch.load(args.backbone_weights, map_location=DEVICE)
        # 检查是完整模型 checkpoint 还是纯 backbone 权重
        if 'model_state_dict' in ckpt:
            # 完整模型 checkpoint → 提取 backbone 部分
            full_sd = ckpt['model_state_dict']
            # 过滤出 backbone 的键
            backbone_sd = {k.replace('backbone.', ''): v
                          for k, v in full_sd.items()
                          if k.startswith('backbone.')}
            print(f"  从模型 checkpoint 加载 backbone ({len(backbone_sd)} 个参数)")
            model.backbone.load_state_dict(backbone_sd)

            # 也加载 emotion head（如果可用）
            emotion_sd = {k.replace('emotion_head.', ''): v
                         for k, v in full_sd.items()
                         if k.startswith('emotion_head.')}
            if len(emotion_sd) > 0:
                print(f"  同时加载 emotion head ({len(emotion_sd)} 个参数)")
                model.emotion_head.load_state_dict(emotion_sd, strict=False)
        else:
            # 纯 backbone 权重
            print(f"  加载 backbone (纯权重): {args.backbone_weights}")
            model.backbone.load_state_dict(ckpt)

    if args.emotion_weights:
        print(f"  加载 emotion head: {args.emotion_weights}")
        ckpt_emo = torch.load(args.emotion_weights, map_location=DEVICE)
        if 'model_state_dict' in ckpt_emo:
            emo_sd = {k.replace('emotion_head.', ''): v
                     for k, v in ckpt_emo['model_state_dict'].items()
                     if k.startswith('emotion_head.')}
            model.emotion_head.load_state_dict(emo_sd, strict=False)
            print(f"  从 checkpoint 提取 emotion head ({len(emo_sd)} 个参数)")
        else:
            model.emotion_head.load_state_dict(ckpt_emo)

    if args.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
        print("  Backbone 已冻结，只训练 head")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)
    print(f"  总参数: {total_params:,}")
    print(f"  可训练: {trainable_params:,}")

    # --- 优化器 ---
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5,
    )

    # Cosine 学习率调度（余弦退火）
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # --- 恢复 checkpoint ---
    start_epoch = 1
    if args.resume:
        print(f"  恢复: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"  从 epoch {start_epoch} 继续训练")

    # --- 训练循环 ---
    best_val_emo_loss = float("inf")
    best_val_genre_loss = float("inf")
    patience_counter = 0

    # 记录
    history = {
        "train_total": [],
        "train_emotion": [],
        "train_genre": [],
        "val_emotion_raw": [],
        "val_genre_raw": [],
        "val_accuracy": [],
        "val_V_mean_R2": [],
        "val_A_mean_R2": [],
    }

    checkpoint_dir = Path(CONFIG["paths"]["checkpoint_dir"])
    log_dir = Path(CONFIG["paths"]["log_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  开始训练 (共 {args.epochs} 轮, 早停耐心 {args.patience})")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        # 训练
        train_losses = train_epoch_multitask(
            model=model,
            deam_loader=train_emo_loader,
            gtzan_loader=train_genre_loader,
            optimizer=optimizer,
            loss_weights=loss_weights,
            device=DEVICE,
            grad_clip=1.0,
            tasks=['emotion', 'genre'],
        )

        # 验证
        val_losses, val_metrics = validate_multitask(
            model=model,
            deam_loader=val_emo_loader,
            gtzan_loader=val_genre_loader,
            loss_weights=loss_weights,
            device=DEVICE,
            tasks=['emotion', 'genre'],
        )

        # 学习率更新
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # 记录
        history["train_total"].append(train_losses["total"])
        history["train_emotion"].append(train_losses["emotion"])
        history["train_genre"].append(train_losses["genre"])
        history["val_emotion_raw"].append(val_losses["emotion_raw"])
        history["val_genre_raw"].append(val_losses["genre_raw"])

        if "accuracy" in val_metrics:
            history["val_accuracy"].append(val_metrics["accuracy"])
        if "V_mean_R2" in val_metrics:
            history["val_V_mean_R2"].append(val_metrics["V_mean_R2"])
        else:
            history["val_V_mean_R2"].append(0.0)
        if "A_mean_R2" in val_metrics:
            history["val_A_mean_R2"].append(val_metrics["A_mean_R2"])
        else:
            history["val_A_mean_R2"].append(0.0)

        # 打印日志
        if epoch % args.log_interval == 0 or epoch == 1:
            current_lr = scheduler.get_last_lr()[0]
            print(f"\n[{epoch_time:.0f}s] Epoch {epoch}/{args.epochs}  (lr={current_lr:.2e})")
            print(f"  Train | total={train_losses['total']:.4f}  "
                  f"emo={train_losses['emotion_raw']:.4f}  "
                  f"genre={train_losses['genre_raw']:.4f}")
            print(f"  Val   | total={val_losses['total']:.4f}  "
                  f"emo={val_losses['emotion_raw']:.4f}  "
                  f"genre={val_losses['genre_raw']:.4f}")

            # 指标
            key_metrics = ["V_mean_R2", "V_mean_RMSE", "V_mean_PearsonR",
                           "A_mean_R2", "A_mean_RMSE", "A_mean_PearsonR",
                           "accuracy", "f1_macro"]
            metric_str = "  |  ".join(
                [f"{k}: {val_metrics[k]:.4f}"
                 for k in key_metrics if k in val_metrics]
            )
            if metric_str:
                print(f"  Metrics: {metric_str}")

        # 保存最佳模型（按 emotion val loss）
        val_emo = val_losses["emotion_raw"]
        val_genre = val_losses["genre_raw"]

        improved = False
        if val_emo < best_val_emo_loss:
            best_val_emo_loss = val_emo
            improved = True

        if val_genre < best_val_genre_loss:
            best_val_genre_loss = val_genre

        if improved:
            patience_counter = 0
            checkpoint_path = checkpoint_dir / "best_multitask_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_emo_loss": val_emo,
                "val_genre_loss": val_genre,
                "metrics": val_metrics,
                "config": CONFIG,
                "args": vars(args),
            }, checkpoint_path)
            print(f"  ✓ 保存最佳模型 → {checkpoint_path} "
                  f"(emo_loss={best_val_emo_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  ⏹ 早停: {args.patience} 轮未改善")
                break

    # --- 完成 ---
    print(f"\n{'='*60}")
    print(f"  训练完成")
    print(f"  最佳验证 emotion loss: {best_val_emo_loss:.4f}")
    print(f"  最佳验证 genre loss:   {best_val_genre_loss:.4f}")
    print(f"  模型: {checkpoint_dir / 'best_multitask_model.pth'}")
    print(f"{'='*60}")

    # 绘制训练曲线
    _plot_multitask_history(
        history, args.epochs,
        save_path=str(checkpoint_dir / "multitask_curves.png"),
    )


def _plot_multitask_history(history: dict, epochs: int, save_path: str):
    """绘制多任务训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 总 loss
    ax = axes[0, 0]
    ax.plot(history["train_total"], label="Train Total", color="blue", alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Total Training Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    # 各任务 loss
    ax = axes[0, 1]
    if history["train_emotion"]:
        ax.plot(history["train_emotion"], label="Train Emotion",
                color="green", alpha=0.7)
    if history["train_genre"]:
        ax.plot(history["train_genre"], label="Train Genre",
                color="orange", alpha=0.7)
    if history["val_emotion_raw"]:
        ax.plot(history["val_emotion_raw"], label="Val Emotion",
                color="green", alpha=0.7, linestyle="--")
    if history["val_genre_raw"]:
        ax.plot(history["val_genre_raw"], label="Val Genre",
                color="orange", alpha=0.7, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Task Losses")
    ax.legend()
    ax.grid(alpha=0.3)

    # 准确率
    ax = axes[1, 0]
    if history["val_accuracy"]:
        ax.plot(history["val_accuracy"], label="Genre Acc",
                color="purple", alpha=0.7)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Genre Classification Accuracy")
        ax.legend()
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No accuracy data",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Genre Accuracy")

    # Emotion R²
    ax = axes[1, 1]
    if history["val_V_mean_R2"]:
        ax.plot(history["val_V_mean_R2"], label="Valence R²",
                color="red", alpha=0.7)
    if history["val_A_mean_R2"]:
        ax.plot(history["val_A_mean_R2"], label="Arousal R²",
                color="blue", alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("R²")
    ax.set_title("Emotion R² Score")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[图表] 训练曲线 → {save_path}")


if __name__ == "__main__":
    main()
