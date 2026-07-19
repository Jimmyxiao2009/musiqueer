"""
MusicUnderstand 2.0 — 训练脚本 (Emotion Head)

Phase 2a: 在 DEAM 上训练情绪理解能力

支持三种训练模式:
  - end_to_end: 从头训练 backbone + emotion head (Step 1 默认)
  - freeze_backbone: 加载预训练 backbone，只训练 head (Phase 2a)
  - finetune: 加载所有权重，联合微调 (Phase 3)

Step 1 特化:
  - 只在 DEAM 上训练
  - 12维度中仅 V/A mean/delta/var 有监督信号 (6/12)
  - delta/var 从标注的 mean 值计算得到，作为 ground truth
"""

import os
import sys
import time
import argparse
from pathlib import Path

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
from data.deam_dataset import create_dataloaders
from utils.metrics import evaluate_emotion

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
        total_loss: 标量
        loss_dict:  各分量 loss 值 (用于日志)
    """
    B, T, D = pred.shape
    mask_3d = mask.unsqueeze(-1).expand_as(pred)  # [B, T, 12]

    # 只计算有效帧的 loss
    diff = (pred - target) ** 2
    diff = diff * mask_3d

    # 各维度独立 mean
    dim_losses = diff.sum(dim=(0, 1)) / (mask_3d.sum(dim=(0, 1)) + 1e-8)

    loss_dict = {
        "V_mean": dim_losses[0],
        "V_delta": dim_losses[1],
        "V_var": dim_losses[2],
        "A_mean": dim_losses[3],
        "A_delta": dim_losses[4],
        "A_var": dim_losses[5],
        "T_mean": dim_losses[6],
        "T_delta": dim_losses[7],
        "T_var": dim_losses[8],
        "C_mean": dim_losses[9],
        "C_delta": dim_losses[10],
        "C_var": dim_losses[11],
    }

    total_loss = (
        dim_losses[0] * loss_weights.get("valence_mean", 1.0) +
        dim_losses[3] * loss_weights.get("arousal_mean", 1.0) +
        dim_losses[1] * loss_weights.get("valence_delta", 0.5) +
        dim_losses[4] * loss_weights.get("arousal_delta", 0.5) +
        dim_losses[2] * loss_weights.get("valence_var", 0.3) +
        dim_losses[5] * loss_weights.get("arousal_var", 0.3) +
        dim_losses[6] * loss_weights.get("tension_mean", 1.0) +
        dim_losses[9] * loss_weights.get("complexity_mean", 1.0) +
        dim_losses[7] * loss_weights.get("tension_delta", 0.5) +
        dim_losses[10] * loss_weights.get("complexity_delta", 0.5) +
        dim_losses[8] * loss_weights.get("tension_var", 0.3) +
        dim_losses[11] * loss_weights.get("complexity_var", 0.3)
    )

    return total_loss, loss_dict


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion,
    loss_weights: dict,
    device: torch.device,
    grad_clip: float = 1.0,
) -> tuple:
    """
    训练一个 epoch

    Returns:
        avg_loss: 平均 loss
        loss_dict: 各分量平均 loss
    """
    model.train()
    total_loss = 0
    total_loss_dict = {k: 0.0 for k in
                       ["V_mean", "V_delta", "V_var",
                        "A_mean", "A_delta", "A_var",
                        "T_mean", "T_delta", "T_var",
                        "C_mean", "C_delta", "C_var"]}
    n_batches = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        mel = batch["mel"].to(device)          # [B, 1, 128, T]
        emotion = batch["emotion"].to(device)  # [B, T, 12]
        mask = batch["mask"].to(device)        # [B, T]

        outputs = model(mel, active_experts=['emotion'])
        pred = outputs['emotion']              # [B, T, 12]

        loss, loss_dict = criterion(
            pred, emotion, mask, loss_weights
        )

        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪（防止 Transformer 梯度爆炸）
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        for k in total_loss_dict:
            total_loss_dict[k] += loss_dict[k].item()
        n_batches += 1

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / n_batches
    avg_loss_dict = {k: v / n_batches for k, v in total_loss_dict.items()}

    return avg_loss, avg_loss_dict


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion,
    loss_weights: dict,
    device: torch.device,
) -> tuple:
    """
    验证集评估

    Returns:
        avg_loss: 平均 loss
        metrics:  评价指标 dict
    """
    model.eval()
    total_loss = 0
    total_loss_dict = {k: 0.0 for k in
                       ["V_mean", "V_delta", "V_var",
                        "A_mean", "A_delta", "A_var",
                        "T_mean", "T_delta", "T_var",
                        "C_mean", "C_delta", "C_var"]}
    n_batches = 0

    all_preds = []
    all_targets = []
    all_masks = []

    for batch in tqdm(loader, desc=" Val", leave=False):
        mel = batch["mel"].to(device)
        emotion = batch["emotion"].to(device)
        mask = batch["mask"].to(device)

        outputs = model(mel, active_experts=['emotion'])
        pred = outputs['emotion']

        loss, loss_dict = criterion(pred, emotion, mask, loss_weights)

        total_loss += loss.item()
        for k in total_loss_dict:
            total_loss_dict[k] += loss_dict[k].item()
        n_batches += 1

        all_preds.append(pred.cpu())
        all_targets.append(emotion.cpu())
        all_masks.append(mask.cpu())

    avg_loss = total_loss / n_batches
    avg_loss_dict = {k: v / n_batches for k, v in total_loss_dict.items()}

    # 计算评价指标
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    metrics = evaluate_emotion(
        all_preds, all_targets, all_masks,
        dim_names=["V_mean", "V_delta", "V_var",
                    "A_mean", "A_delta", "A_var",
                    "T_mean", "T_delta", "T_var",
                    "C_mean", "C_delta", "C_var"],
    )

    return avg_loss, avg_loss_dict, metrics


def plot_training(
    train_losses: list,
    val_losses: list,
    component_losses: dict,
    save_path: str = "training_curves.png",
):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 总 loss
    ax = axes[0, 0]
    ax.plot(train_losses, label="Train", color="blue", alpha=0.7)
    ax.plot(val_losses, label="Val", color="red", alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Total Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    # 各维度 loss
    ax = axes[0, 1]
    for name, values in component_losses.items():
        if len(values) > 0:
            ax.plot(values, label=name, alpha=0.7, linewidth=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Component Losses (Val)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # R² scores
    ax = axes[1, 0]
    r2_keys = [k for k in component_losses.keys()
               if k.endswith("_R2") and len(component_losses[k]) > 0]
    for name in r2_keys:
        ax.plot(component_losses[name], label=name, alpha=0.7, linewidth=1)
    if r2_keys:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("R²")
        ax.set_title("R² Score")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "R² data will appear during training",
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title("R² Score")

    # Pearson r
    ax = axes[1, 1]
    r_keys = [k for k in component_losses.keys()
              if k.endswith("_PearsonR") and len(component_losses[k]) > 0]
    for name in r_keys:
        ax.plot(component_losses[name], label=name, alpha=0.7, linewidth=1)
    if r_keys:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Pearson r")
        ax.set_title("Pearson Correlation")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Pearson r data will appear during training",
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Pearson Correlation")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[图表] 训练曲线 → {save_path}")


def print_metrics(epoch: int, train_loss: float, val_loss: float,
                   loss_dict: dict, metrics: dict, prefix: str = ""):
    """打印训练状态"""
    print(f"\n{prefix}Epoch {epoch}")
    print(f"  Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}")

    # loss 分量
    loss_str = "  ".join(
        [f"{k}: {v:.4f}" for k, v in loss_dict.items()]
    )
    print(f"  Losses: {loss_str}")

    # 指标（只打印几个关键项）
    key_metrics = ["V_mean_R2", "V_mean_RMSE", "V_mean_PearsonR",
                   "A_mean_R2", "A_mean_RMSE", "A_mean_PearsonR",
                   "T_mean_R2", "T_mean_PearsonR",
                   "C_mean_R2", "C_mean_PearsonR"]
    for k in key_metrics:
        if k in metrics:
            print(f"  {k}: {metrics[k]:.4f}", end="  ")
    print()


def main():
    parser = argparse.ArgumentParser(description="MusicUnderstand 2.0 训练")
    parser.add_argument("--mode", type=str, default="end_to_end",
                        choices=["end_to_end", "freeze_backbone", "finetune"],
                        help="训练模式")
    parser.add_argument("--backbone_weights", type=str, default=None,
                        help="预训练 backbone 权重路径")
    parser.add_argument("--emotion_weights", type=str, default=None,
                        help="预训练 emotion head 权重路径")
    parser.add_argument("--resume", type=str, default=None,
                        help="从 checkpoint 恢复训练（含optimizer/epoch，完整状态匹配）")
    parser.add_argument("--init_from", type=str, default=None,
                        help="从已有checkpoint初始化backbone+emotion_head权重"
                             "(strict=False，忽略mood_head等本脚本没有的模块)，"
                             "optimizer从头开始，用于VATC四维训练接着V/A checkpoint微调")
    parser.add_argument("--checkpoint_prefix", type=str, default="best",
                        help="checkpoint文件名前缀，如 vatc → vatc_best_model.pth")
    parser.add_argument("--epochs", type=int, default=None,
                        help="覆盖配置中的 epoch 数")
    args = parser.parse_args()

    print("=" * 60)
    print("   MusicUnderstand 2.0 — 训练启动")
    print(f"   模式: {args.mode}")
    print(f"   设备: {DEVICE}")
    print("=" * 60)

    # --- 加载配置 ---
    model_config = CONFIG["model"]
    train_config = CONFIG["train"]["emotion"]
    data_config = CONFIG["data"]

    epochs = args.epochs or train_config["epochs"]
    lr = train_config["learning_rate"]
    batch_size = CONFIG["train"]["batch_size"]
    loss_weights = train_config["loss_weights"]
    grad_clip = train_config.get("gradient_clip", 1.0)
    patience = train_config["patience"]
    log_interval = train_config["log_interval"]

    seed = CONFIG["train"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- 数据 ---
    print("\n[数据] 加载...")
    train_loader, val_loader, train_dataset, val_dataset = create_dataloaders(
        root_dir=CONFIG["paths"]["deam_root"],
        batch_size=batch_size,
        num_workers=CONFIG["train"]["num_workers"],
    )

    # --- 模型 ---
    print("\n[模型] 初始化...")
    # 弹出配置中不被模型接受的字段
    backbone_cfg = {k: v for k, v in model_config["backbone"].items()
                    if k != "type"}
    emotion_cfg = {k: v for k, v in model_config["experts"]["emotion"].items()
                   if k not in ("loss_weight",)}

    model = MusicUnderstand2(
        backbone_config=backbone_cfg,
        emotion_config=emotion_cfg,
        # mood_head本脚本不训练/不使用(active_experts=['emotion'])，但num_tags要跟
        # --init_from的checkpoint(joint_best_corr_model.pth用MTG的59个标签)对齐，
        # 否则strict=False也会因为shape不匹配报错(strict=False只忽略缺失/多余键，
        # 不忽略同名键的shape冲突)
        mood_config={"num_tags": 59},
    ).to(DEVICE)

    if args.mode == "freeze_backbone" and args.backbone_weights:
        print(f"  加载 backbone: {args.backbone_weights}")
        model.backbone.load_state_dict(
            torch.load(args.backbone_weights, map_location=DEVICE))
        # 冻结 backbone
        for p in model.backbone.parameters():
            p.requires_grad = False
        print("  Backbone 已冻结，只训练 Emotion Head")

    if args.mode == "finetune":
        if args.backbone_weights:
            model.backbone.load_state_dict(
                torch.load(args.backbone_weights, map_location=DEVICE))
        if args.emotion_weights:
            model.emotion_head.load_state_dict(
                torch.load(args.emotion_weights, map_location=DEVICE))
        print("  Backbone + Emotion Head 权重已加载，联合微调")

    if args.init_from:
        print(f"\n  [初始化] 从 {args.init_from} 加载 backbone+emotion_head (strict=False)")
        ckpt = torch.load(args.init_from, map_location=DEVICE, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"    缺失(本脚本没有的模块,正常): {len(missing)}个  "
              f"多余(忽略,如mood_head): {len(unexpected)}个")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)
    print(f"  总参数: {total_params:,}")
    print(f"  可训练: {trainable_params:,}")

    # --- 优化器 ---
    if args.mode == "end_to_end":
        # AdamW（比 Adam 更优，weight decay 更干净）
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=train_config.get("weight_decay", 1e-5),
        )
    else:
        # Frozen backbone 模式用 Adam，lr 大一点
        optimizer = optim.Adam(
            model.parameters(), lr=lr)

    # Cosine 学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

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
    best_val_loss = float("inf")
    best_corr_score = float("-inf")
    patience_counter = 0
    train_losses = []
    val_losses = []
    component_losses = {}  # 存储各分量/指标的 epoch 序列

    checkpoint_dir = Path(CONFIG["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  开始训练 (共 {epochs} 轮, 早停耐心 {patience})")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()

        # 训练
        train_loss, _ = train_epoch(
            model, train_loader, optimizer,
            build_emotion_loss, loss_weights, DEVICE, grad_clip,
        )
        train_losses.append(train_loss)

        # 验证
        val_loss, val_loss_dict, val_metrics = validate(
            model, val_loader,
            build_emotion_loss, loss_weights, DEVICE,
        )
        val_losses.append(val_loss)

        # 记录分量 loss
        for k, v in val_loss_dict.items():
            if k not in component_losses:
                component_losses[k] = []
            component_losses[k].append(v)

        # 记录指标
        for k, v in val_metrics.items():
            if k not in component_losses:
                component_losses[k] = []
            component_losses[k].append(v)

        # 学习率更新
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # 打印日志
        if epoch % log_interval == 0 or epoch == 1:
            print_metrics(
                epoch, train_loss, val_loss, val_loss_dict, val_metrics,
                prefix=f"[{epoch_time:.0f}s] ",
            )

        # 保存最佳相关性模型 (按 V/A PearsonR 均值，代表下游可用性；
        # val_loss 最低的 epoch 不一定是相关性最高的 epoch，两者独立保存。
        # 必须在早停 break 之前执行，否则触发早停的最后一个 epoch 会被跳过检查)
        corr_score = (
            val_metrics.get("V_mean_PearsonR", 0.0)
            + val_metrics.get("A_mean_PearsonR", 0.0)
        ) / 2
        if corr_score > best_corr_score:
            best_corr_score = corr_score
            corr_checkpoint_path = checkpoint_dir / f"{args.checkpoint_prefix}_corr_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "corr_score": corr_score,
                "metrics": val_metrics,
                "config": CONFIG,
            }, corr_checkpoint_path)
            print(f"  ✓ 保存最佳相关性模型 (V+A PearsonR均值) → {corr_checkpoint_path} "
                  f"(corr_score={best_corr_score:.4f})")

        # 保存最佳模型 (按 val_loss，用于早停判断)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint_path = checkpoint_dir / f"{args.checkpoint_prefix}_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "metrics": val_metrics,
                "config": CONFIG,
            }, checkpoint_path)
            print(f"  ✓ 保存最佳模型 (val_loss) → {checkpoint_path} "
                  f"(val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  ⏹ 早停: {patience} 轮未改善")
                break

    # --- 完成 ---
    print(f"\n{'='*60}")
    print(f"  训练完成")
    print(f"  最佳验证 loss: {best_val_loss:.4f}")
    print(f"  模型: {checkpoint_dir / (args.checkpoint_prefix + '_model.pth')}")
    print(f"{'='*60}")

    # 绘制训练曲线
    plot_training(train_losses, val_losses, component_losses,
                  save_path=str(checkpoint_dir / f"{args.checkpoint_prefix}_training_curves.png"))


if __name__ == "__main__":
    main()
