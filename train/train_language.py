"""
MusicUnderstand 3.0 — Language Decoder 训练脚本
Training Script for Language Decoder

Stages / 训练阶段:
  Stage 1 (当前): 冻结 backbone，训练 decoder
  Stage 2 (未来): 联合微调 backbone + decoder

训练策略:
  - Teacher forcing with cross-entropy loss
  - AdamW + CosineAnnealingLR
  - 支持从 checkpoint 恢复训练
  - 日志: loss, perplexity, learning rate
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Optional, Dict

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# 确保能找到项目模块
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

from model.music_understand import MusicUnderstand2
from model.experts.language_decoder import LanguageDecoder
from data.music_caption_dataset import create_caption_dataloaders

# --- 配置 Config ---
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "config", "default.yaml")
with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_loss(
    logits: torch.Tensor,          # [B, T, vocab_size]
    input_ids: torch.Tensor,       # [B, T]
    attention_mask: torch.Tensor,  # [B, T]
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    计算语言建模损失 (Cross-entropy with optional label smoothing)

    Args:
        logits: decoder 输出的 logits
        input_ids: 目标 token IDs
        attention_mask: padding mask (1=有效, 0=padding)
        label_smoothing: 标签平滑系数 (默认 0.0)

    Returns:
        loss: scalar
    """
    B, T, V = logits.shape

    # Shift: predict next_token given current token
    # logits[:, :-1, :]  → predict token at position 1..T-1
    # input_ids[:, 1:]   → target at position 1..T-1
    shift_logits = logits[:, :-1, :].contiguous()       # [B, T-1, V]
    shift_labels = input_ids[:, 1:].contiguous()         # [B, T-1]
    shift_mask = attention_mask[:, 1:].contiguous()      # [B, T-1]

    loss_fct = nn.CrossEntropyLoss(
        reduction='none',
        label_smoothing=label_smoothing,
    )

    loss = loss_fct(
        shift_logits.view(-1, V),    # [(B*(T-1)), V]
        shift_labels.view(-1),       # [(B*(T-1))]
    )  # [(B*(T-1))]

    # Apply mask and reduce
    loss = loss.view(B, -1) * shift_mask  # [B, T-1]
    loss = loss.sum() / shift_mask.sum()

    return loss


def compute_perplexity(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> float:
    """
    计算困惑度 (Perplexity = exp(cross_entropy))

    Args:
        logits: decoder logits [B, T, V]
        input_ids: target IDs [B, T]
        attention_mask: padding mask [B, T]

    Returns:
        perplexity: scalar
    """
    loss = compute_loss(logits, input_ids, attention_mask)
    return float(torch.exp(loss).item())


def get_tokenizer():
    """
    加载 GPT-2 tokenizer

    Returns:
        tokenizer: AutoTokenizer from transformers
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError(
            "Please install transformers: pip install transformers"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "tokenizers", "gpt2"),
        local_files_only=True,
    )
    # GPT-2 doesn't have a pad token by default
    # We use <|endoftext|> (50256) as pad token
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def train_epoch(
    model: nn.Module,
    decoder: LanguageDecoder,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    epoch: int,
    num_epochs: int,
    gradient_clip: float = 1.0,
    log_interval: int = 10,
    label_smoothing: float = 0.0,
) -> Dict[str, float]:
    """
    训练一个 epoch

    Returns:
        metrics: {'loss', 'perplexity', 'lr'}
    """
    model.eval()   # backbone in eval mode (frozen)
    decoder.train()

    total_loss = 0.0
    total_tokens = 0
    batch_count = 0
    start_time = time.time()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

    for batch_idx, batch in enumerate(pbar):
        memory = batch["memory"].to(DEVICE)               # [B, T_audio, d_model]
        input_ids = batch["input_ids"].to(DEVICE)          # [B, T_text]
        attention_mask = batch["attention_mask"].to(DEVICE) # [B, T_text]

        # Forward
        logits = decoder(
            memory=memory,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Loss
        loss = compute_loss(logits, input_ids, attention_mask, label_smoothing)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), gradient_clip)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Stats
        batch_loss = loss.item()
        total_loss += batch_loss * attention_mask[:, 1:].sum().item()
        total_tokens += attention_mask[:, 1:].sum().item()
        batch_count += 1

        # Log
        if batch_idx % log_interval == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            perplexity = np.exp(batch_loss)
            pbar.set_postfix({
                "loss": f"{batch_loss:.4f}",
                "ppl": f"{perplexity:.2f}",
                "lr": f"{current_lr:.2e}",
            })

    avg_loss = total_loss / max(total_tokens, 1)
    avg_perplexity = np.exp(avg_loss)

    return {
        "loss": avg_loss,
        "perplexity": avg_perplexity,
        "lr": optimizer.param_groups[0]["lr"],
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    decoder: LanguageDecoder,
    val_loader: DataLoader,
) -> Dict[str, float]:
    """
    Validation loop / 验证循环

    Returns:
        metrics: {'loss', 'perplexity'}
    """
    model.eval()
    decoder.eval()

    total_loss = 0.0
    total_tokens = 0

    for batch in tqdm(val_loader, desc="[Valid]"):
        memory = batch["memory"].to(DEVICE)
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        logits = decoder(memory, input_ids, attention_mask)
        loss = compute_loss(logits, input_ids, attention_mask)

        total_loss += loss.item() * attention_mask[:, 1:].sum().item()
        total_tokens += attention_mask[:, 1:].sum().item()

    avg_loss = total_loss / max(total_tokens, 1)
    avg_perplexity = np.exp(avg_loss)

    return {"loss": avg_loss, "perplexity": avg_perplexity}


def save_checkpoint(
    decoder: LanguageDecoder,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_dir: str,
    filename: str = "language_decoder_checkpoint.pth",
):
    """保存 checkpoint"""
    checkpoint = {
        "epoch": epoch,
        "decoder_state_dict": decoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
    }
    path = os.path.join(checkpoint_dir, filename)
    torch.save(checkpoint, path)
    print(f"[Checkpoint] Saved to {path}")


def load_checkpoint(
    decoder: LanguageDecoder,
    optimizer: Optional[optim.Optimizer],
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    checkpoint_path: str,
) -> int:
    """加载 checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"[Checkpoint] No checkpoint found at {checkpoint_path}, starting from scratch")
        return 0

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint.get("epoch", 0) + 1
    print(f"[Checkpoint] Loaded from {checkpoint_path} (epoch {checkpoint.get('epoch', 0)})")
    return start_epoch


def main():
    parser = argparse.ArgumentParser(description="Train Language Decoder")
    parser.add_argument("--config", type=str, default=_CONFIG_PATH,
                        help="Path to config YAML")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (overrides config)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Path to resume checkpoint")
    parser.add_argument("--freeze-backbone", action="store_true",
                        default=True,
                        help="Freeze backbone parameters")
    parser.add_argument("--use-musiccaps", action="store_true",
                        default=True,
                        help="Use MusicCaps dataset")
    parser.add_argument("--use-synthetic", action="store_true",
                        default=True,
                        help="Use synthetic caption data")
    parser.add_argument("--max-samples", type=int, default=1000,
                        help="Max samples to use (for quick testing)")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Label smoothing coefficient")
    parser.add_argument("--gradient-clip", type=float, default=1.0,
                        help="Gradient clipping norm")
    parser.add_argument("--warmup-steps", type=int, default=1000,
                        help="LR warmup steps")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Log every N batches")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"  MusicUnderstand 3.0 — Language Decoder Training")
    print(f"  Device: {DEVICE}")
    print(f"{'='*60}")

    # --- 加载 Tokenizer ---
    print("\n[1/6] Loading GPT-2 tokenizer...")
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.vocab_size
    print(f"  Vocab size: {vocab_size}")

    # --- 构建模型 ---
    print("\n[2/6] Building model...")
    # 加载 Step 1 checkpoint (含 backbone + emotion head)
    checkpoint_dir = CONFIG["paths"]["checkpoint_dir"]
    step1_checkpoint = os.path.join(checkpoint_dir, "best_model.pth")

    # 先构建 backbone 用于预计算 features
    from model.backbone import MusicTransformerBackbone
    backbone = MusicTransformerBackbone(
        d_model=256, nhead=8, num_layers=6, dim_feedforward=1024,
    ).to(DEVICE)

    # 构建完整模型（包含 backbone + 可选 head）
    model = MusicUnderstand2().to(DEVICE)

    if os.path.exists(step1_checkpoint):
        print(f"  Loading Step 1 checkpoint: {step1_checkpoint}")
        checkpoint = torch.load(step1_checkpoint, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        # Also load backbone weights into standalone backbone
        backbone.load_state_dict(model.backbone.state_dict())
        print(f"  Step 1 checkpoint loaded successfully")
    else:
        print(f"  Warning: Step 1 checkpoint not found at {step1_checkpoint}")
        print(f"  Using randomly initialized backbone")

    # --- 构建 Decoder ---
    decoder = LanguageDecoder(
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        vocab_size=vocab_size,
        max_seq_len=128,
    ).to(DEVICE)

    print(f"  Backbone params: {sum(p.numel() for p in model.backbone.parameters()):,}")
    print(f"  Decoder params:  {sum(p.numel() for p in decoder.parameters()):,}")

    # --- 冻结 backbone ---
    if args.freeze_backbone:
        print("\n[3/6] Freezing backbone...")
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in backbone.parameters():
            param.requires_grad = False

    # --- 创建 DataLoaders ---
    print("\n[4/6] Creating caption dataloaders...")
    train_loader, val_loader = create_caption_dataloaders(
        backbone=backbone if args.use_musiccaps or args.use_synthetic else None,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=0,  # Windows safe
        use_musiccaps=False,
        use_synthetic=True,
        max_samples_musiccaps=args.max_samples,
        max_samples_synthetic=args.max_samples,
        max_text_len=64,
        cache_dir="./data/caption_cache",
        device=str(DEVICE),
    )

    print(f"  Train samples: {len(train_loader.dataset)}")
    if val_loader:
        print(f"  Val samples:   {len(val_loader.dataset)}")

    # --- Optimizer & Scheduler ---
    print("\n[5/6] Setting up optimizer and scheduler...")
    optimizer = optim.AdamW(
        decoder.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.999),
    )

    # Total training steps
    total_steps = len(train_loader) * args.epochs

    # Cosine annealing with linear warmup
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=args.lr * 0.01,
    )

    print(f"  Optimizer: AdamW (lr={args.lr}, weight_decay=0.01)")
    print(f"  Scheduler: CosineAnnealing (T_max={total_steps})")
    print(f"  Total steps: {total_steps}")

    # --- 恢复 checkpoint ---
    start_epoch = 0
    if args.checkpoint:
        start_epoch = load_checkpoint(decoder, optimizer, scheduler, args.checkpoint)

    # --- Training Loop ---
    print(f"\n[6/6] Starting training...")
    print(f"{'='*60}")

    best_val_ppl = float('inf')
    patience_counter = 0
    patience = 10

    for epoch in range(start_epoch, args.epochs):
        # Train
        train_metrics = train_epoch(
            model, decoder, train_loader, optimizer, scheduler,
            epoch, args.epochs,
            gradient_clip=args.gradient_clip,
            log_interval=args.log_interval,
            label_smoothing=args.label_smoothing,
        )

        # Validate
        if val_loader is not None:
            val_metrics = validate(model, decoder, val_loader)
            val_ppl = val_metrics["perplexity"]

            print(
                f"\nEpoch {epoch+1:3d}/{args.epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train PPL: {train_metrics['perplexity']:.2f} | "
                f"Val PPL: {val_ppl:.2f} | "
                f"LR: {train_metrics['lr']:.2e}"
            )

            # Save best model
            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                save_checkpoint(
                    decoder, optimizer, scheduler, epoch,
                    {"train_loss": train_metrics["loss"],
                     "val_loss": val_metrics["loss"],
                     "val_ppl": val_ppl},
                    checkpoint_dir,
                    "language_decoder_best.pth",
                )
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[Early Stopping] No improvement for {patience} epochs")
                    break
        else:
            print(
                f"\nEpoch {epoch+1:3d}/{args.epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train PPL: {train_metrics['perplexity']:.2f} | "
                f"LR: {train_metrics['lr']:.2e}"
            )

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                decoder, optimizer, scheduler, epoch,
                train_metrics,
                checkpoint_dir,
                f"language_decoder_epoch_{epoch+1}.pth",
            )

    # Save final checkpoint
    save_checkpoint(
        decoder, optimizer, scheduler, args.epochs - 1,
        {"train_loss": train_metrics["loss"]},
        checkpoint_dir,
        "language_decoder_final.pth",
    )

    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    if val_loader:
        print(f"  Best Validation Perplexity: {best_val_ppl:.2f}")
    print(f"  Checkpoints saved to: {checkpoint_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
