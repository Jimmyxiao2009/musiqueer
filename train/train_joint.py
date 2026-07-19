"""
联合训练: DEAM (Valence/Arousal回归) + MTG-Jamendo mood/theme (多标签分类)

动机: DEAM单独训练时Valence学不起来(R²≈0，见docs/PROGRESS.md "Step 1 重训会话")。
假设: MTG-Jamendo的56个mood/theme标签(happy/sad/dark/uplifting...)带valence语义，
用它做辅助多标签分类任务、和VA回归共享backbone联合训练，可能让共享表征学到更多
valence相关线索，帮VA回归"输血"。这是本次会话要验证的假设，不保证一定有效——
如果联合训练后Valence R²没有改善，说明问题不在"数据量"而在别处（比如需要真正的
语义/歌词信息，纯音频不够），要如实记录，不要为了让结果看起来更好而选择性汇报。

训练方式: 每个step同时过一个DEAM batch(算VA loss)和一个MTG batch(算BCE loss)，
两个loss加权求和后一次backward，共享的backbone参数会同时受两个任务的梯度影响。
两个loader长度不同时，短的那个循环取(itertools.cycle)。

用法:
    python train/train_joint.py --epochs 25 --mood_weight 0.3 \
        --init_from ../checkpoints/best_model_run1_epoch4.pth
"""

import os
import sys
import time
import argparse
import itertools
from pathlib import Path

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

from model.music_understand import MusicUnderstand2
from data.deam_dataset import create_dataloaders
from data.mtg_jamendo import create_mood_dataloaders, CACHE_DIR
from utils.metrics import evaluate_emotion
from train_emotion import build_emotion_loss

_CONFIG_PATH = os.path.join(_root, "config", "default.yaml")
with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mood_accuracy(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> dict:
    """简单多标签指标：micro-F1 + 平均每样本命中率"""
    preds = (torch.sigmoid(logits) > threshold).float()
    tp = (preds * labels).sum()
    fp = (preds * (1 - labels)).sum()
    fn = ((1 - preds) * labels).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {"mood_micro_f1": f1.item(), "mood_precision": precision.item(),
            "mood_recall": recall.item()}


def run_epoch(
    model, deam_loader, mtg_loader, optimizer, loss_weights, mood_weight,
    device, grad_clip, train: bool, skip_steps: int = 0, checkpoint_fn=None,
    checkpoint_every: int = 60, mood_pos_weight=None,
):
    """
    skip_steps: 从头跳过多少步（配合checkpoint_fn实现epoch内断点续训）
    checkpoint_fn: (step_idx) -> None，训练时每 checkpoint_every 步调用一次，
        用于比"每个epoch"更高频地落盘（一个epoch要~45-50min，比后台任务被
        kill的间隔还长，必须epoch内也能存档，否则可能永远存不下第一个checkpoint）
    """
    model.train() if train else model.eval()

    n_deam, n_mtg = len(deam_loader), len(mtg_loader)
    n_steps = max(n_deam, n_mtg)
    deam_iter = itertools.cycle(deam_loader) if n_deam < n_steps else iter(deam_loader)
    mtg_iter = itertools.cycle(mtg_loader) if n_mtg < n_steps else iter(mtg_loader)

    total_emotion_loss, total_mood_loss = 0.0, 0.0
    all_preds, all_targets, all_masks = [], [], []
    all_mood_logits, all_mood_labels = [], []

    if skip_steps > 0:
        for _ in range(skip_steps):
            next(deam_iter)
            next(mtg_iter)

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        pbar = tqdm(range(skip_steps, n_steps), desc="Train" if train else " Val", leave=False)
        for step_idx in pbar:
            deam_batch = next(deam_iter)
            mtg_batch = next(mtg_iter)

            mel_d = deam_batch["mel"].to(device)
            emotion = deam_batch["emotion"].to(device)
            mask = deam_batch["mask"].to(device)

            mel_m = mtg_batch["mel"].to(device)
            mood_label = mtg_batch["label"].to(device)

            out_d = model(mel_d, active_experts=["emotion"])
            emotion_loss, _ = build_emotion_loss(out_d["emotion"], emotion, mask, loss_weights)

            out_m = model(mel_m, active_experts=["mood"])
            mood_loss = F.binary_cross_entropy_with_logits(
                out_m["mood_logits"], mood_label, pos_weight=mood_pos_weight)

            total_loss = emotion_loss + mood_weight * mood_loss

            if train:
                optimizer.zero_grad()
                total_loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_emotion_loss += emotion_loss.item()
            total_mood_loss += mood_loss.item()

            all_preds.append(out_d["emotion"].detach().cpu())
            all_targets.append(emotion.cpu())
            all_masks.append(mask.cpu())
            all_mood_logits.append(out_m["mood_logits"].detach().cpu())
            all_mood_labels.append(mood_label.cpu())

            pbar.set_postfix({"emo": f"{emotion_loss.item():.4f}",
                              "mood": f"{mood_loss.item():.4f}"})

            if train and checkpoint_fn is not None and (step_idx + 1) % checkpoint_every == 0:
                checkpoint_fn(step_idx + 1)

    n_done = n_steps - skip_steps
    avg_emotion_loss = total_emotion_loss / n_done
    avg_mood_loss = total_mood_loss / n_done

    metrics = evaluate_emotion(
        torch.cat(all_preds), torch.cat(all_targets), torch.cat(all_masks),
        dim_names=["V_mean", "V_delta", "V_var", "A_mean", "A_delta", "A_var",
                   "T_mean", "T_delta", "T_var", "C_mean", "C_delta", "C_var"],
    )
    metrics.update(mood_accuracy(torch.cat(all_mood_logits), torch.cat(all_mood_labels)))

    return avg_emotion_loss, avg_mood_loss, metrics


def print_metrics(epoch, emo_loss, mood_loss, metrics, prefix=""):
    print(f"\n{prefix}Epoch {epoch}")
    print(f"  Emotion Loss: {emo_loss:.4f}  Mood Loss: {mood_loss:.4f}")
    # 打印所有情绪维度 (V/A/T/C × mean/delta/var)
    dim_order = ["V_mean", "V_delta", "V_var",
                 "A_mean", "A_delta", "A_var",
                 "T_mean", "T_delta", "T_var",
                 "C_mean", "C_delta", "C_var"]
    for dim in dim_order:
        r2 = metrics.get(f"{dim}_R2")
        pr = metrics.get(f"{dim}_PearsonR")
        ccc = metrics.get(f"{dim}_CCC")
        if r2 is not None:
            print(f"  {dim}: R2={r2:.4f}  r={pr:.4f}  CCC={ccc:.4f}")
    if "mood_micro_f1" in metrics:
        print(f"  mood_micro_f1: {metrics['mood_micro_f1']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--mood_weight", type=float, default=0.3,
                        help="mood loss相对emotion loss的权重")
    parser.add_argument("--init_from", type=str, default=None,
                        help="从已有checkpoint初始化backbone+emotion_head (strict=False)")
    parser.add_argument("--resume", type=str, default=None,
                        help="从本脚本自己存的checkpoint完整恢复(含optimizer/epoch，"
                             "支持epoch内的joint_latest.pth，也支持epoch末尾存的checkpoint)")
    parser.add_argument("--checkpoint_every", type=int, default=60,
                        help="训练时每多少step存一次epoch内断点(joint_latest.pth)。"
                             "一个epoch~425step/45-50min，比后台任务常见的被kill间隔"
                             "还长，必须epoch内也能存档")
    parser.add_argument("--mtg_cache_dir", type=str, default=CACHE_DIR)
    args = parser.parse_args()

    print("=" * 60)
    print("   联合训练: DEAM (VA) + MTG-Jamendo (mood/theme)")
    print(f"   设备: {DEVICE}  mood_weight={args.mood_weight}")
    print("=" * 60)

    model_config = CONFIG["model"]
    train_config = CONFIG["train"]["emotion"]
    batch_size = CONFIG["train"]["batch_size"]
    num_workers = CONFIG["train"]["num_workers"]
    loss_weights = train_config["loss_weights"]
    grad_clip = train_config.get("gradient_clip", 1.0)
    patience = train_config["patience"]

    seed = CONFIG["train"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("\n[数据] 加载 DEAM...")
    deam_train, deam_val, _, _ = create_dataloaders(
        root_dir=CONFIG["paths"]["deam_root"], batch_size=batch_size,
        num_workers=num_workers,
    )
    print("\n[数据] 加载 MTG-Jamendo mood/theme (增量缓存)...")
    mtg_train, mtg_val, _, _, tags, mood_pos_weight = create_mood_dataloaders(
        cache_dir=args.mtg_cache_dir, batch_size=batch_size, num_workers=num_workers,
    )
    mood_pos_weight = mood_pos_weight.to(DEVICE)

    backbone_cfg = {k: v for k, v in model_config["backbone"].items() if k != "type"}
    emotion_cfg = {k: v for k, v in model_config["experts"]["emotion"].items()
                   if k not in ("loss_weight",)}
    mood_cfg = {"num_tags": len(tags)}

    model = MusicUnderstand2(
        backbone_config=backbone_cfg, emotion_config=emotion_cfg, mood_config=mood_cfg,
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=train_config["learning_rate"],
                            weight_decay=train_config.get("weight_decay", 1e-5))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 1
    best_val_loss = float("inf")
    best_corr_score = float("-inf")
    patience_counter = 0

    if args.init_from:
        print(f"\n[初始化] 从 {args.init_from} 加载 backbone+emotion_head (strict=False)")
        ckpt = torch.load(args.init_from, map_location=DEVICE, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"  缺失(新增模块,正常): {len(missing)}个  多余(忽略): {len(unexpected)}个")

    checkpoint_dir = Path(CONFIG["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    skip_steps = 0
    if args.resume:
        print(f"\n[恢复] {args.resume}")
        ckpt = torch.load(args.resume, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("step"):
            # epoch内断点：同一epoch继续，跳过已完成的step数
            # (跳过的是"数量"而非"完全相同的样本"——DEAM train_loader shuffle=True，
            #  重新迭代会重新打乱，SGD不要求严格重放同一批次，只要覆盖量大致对得上)
            start_epoch = ckpt["epoch"]
            skip_steps = ckpt["step"]
            print(f"  epoch {start_epoch} 内恢复，跳过前 {skip_steps} step")
        else:
            start_epoch = ckpt["epoch"] + 1
            print(f"  从 epoch {start_epoch} 继续")
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)
        best_corr_score = ckpt.get("best_corr_score", best_corr_score)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[模型] 总参数: {total_params:,} (含mood_head: {len(tags)}个标签)")

    print(f"\n{'='*60}\n  开始联合训练 (共 {args.epochs} 轮, 早停耐心 {patience})\n{'='*60}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        def _save_mid_epoch(step_idx, _epoch=epoch):
            torch.save({
                "epoch": _epoch, "step": step_idx,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss, "best_corr_score": best_corr_score,
                "config": CONFIG, "tags": tags,
            }, checkpoint_dir / "joint_latest.pth")
            print(f"  [epoch内存档] step {step_idx} → joint_latest.pth")

        train_emo_loss, train_mood_loss, _ = run_epoch(
            model, deam_train, mtg_train, optimizer, loss_weights,
            args.mood_weight, DEVICE, grad_clip, train=True,
            skip_steps=skip_steps, checkpoint_fn=_save_mid_epoch,
            checkpoint_every=args.checkpoint_every, mood_pos_weight=mood_pos_weight,
        )
        skip_steps = 0  # 只在恢复后的第一个epoch生效一次
        val_emo_loss, val_mood_loss, val_metrics = run_epoch(
            model, deam_val, mtg_val, optimizer, loss_weights,
            args.mood_weight, DEVICE, grad_clip, train=False,
            mood_pos_weight=mood_pos_weight,
        )
        scheduler.step()

        val_loss = val_emo_loss + args.mood_weight * val_mood_loss
        elapsed = time.time() - t0

        print_metrics(epoch, val_emo_loss, val_mood_loss, val_metrics,
                      prefix=f"[{elapsed:.0f}s] ")

        corr_score = (val_metrics.get("V_mean_PearsonR", 0.0)
                     + val_metrics.get("A_mean_PearsonR", 0.0)) / 2
        if corr_score > best_corr_score:
            best_corr_score = corr_score
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss, "corr_score": corr_score,
                "metrics": val_metrics, "config": CONFIG, "tags": tags,
            }, checkpoint_dir / "joint_best_corr_model.pth")
            print(f"  * 保存最佳相关性模型 (corr_score={best_corr_score:.4f})")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss, "metrics": val_metrics,
                "config": CONFIG, "tags": tags,
            }, checkpoint_dir / "joint_best_model.pth")
            print(f"  * 保存最佳模型 (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  早停: {patience} 轮未改善")
                break

    print(f"\n{'='*60}\n  联合训练完成\n  最佳val_loss={best_val_loss:.4f}  最佳corr_score={best_corr_score:.4f}\n{'='*60}")


if __name__ == "__main__":
    main()
