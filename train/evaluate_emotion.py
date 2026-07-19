"""
评估指定 checkpoint 在 DEAM 验证集上的情绪回归效果。

用法:
    python train/evaluate_emotion.py --checkpoint checkpoints/best_model_run1_epoch4.pth
"""

import os
import sys
import argparse

import yaml
import torch

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

from model.music_understand import MusicUnderstand2
from data.deam_dataset import create_dataloaders
from utils.metrics import evaluate_emotion

_CONFIG_PATH = os.path.join(_root, "config", "default.yaml")
with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    print(f"设备: {DEVICE}")
    print(f"Checkpoint: {args.checkpoint}")

    model_config = CONFIG["model"]
    backbone_cfg = {k: v for k, v in model_config["backbone"].items() if k != "type"}
    emotion_cfg = {k: v for k, v in model_config["experts"]["emotion"].items()
                   if k not in ("loss_weight",)}

    model = MusicUnderstand2(
        backbone_config=backbone_cfg,
        emotion_config=emotion_cfg,
        # num_tags要跟joint_*/vatc_*系列checkpoint对齐；strict=False兼容更早期
        # 完全没有mood_head的checkpoint(如best_model_run1_epoch4.pth)
        mood_config={"num_tags": 59},
    ).to(DEVICE)

    ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"  [提示] 缺失键{len(missing)}个/多余键{len(unexpected)}个"
              f"(通常是mood_head在新旧checkpoint间的差异，不影响emotion_head评估)")
    model.eval()
    print(f"  加载成功 (epoch={ckpt.get('epoch', '?')}, "
          f"val_loss={ckpt.get('val_loss', '?')})")

    print("\n[数据] 加载 DEAM 验证集...")
    _, val_loader, _, val_dataset = create_dataloaders(
        root_dir=CONFIG["paths"]["deam_root"],
        batch_size=CONFIG["train"]["batch_size"],
        num_workers=0,
    )
    print(f"  验证集: {len(val_dataset)} 段")

    all_preds, all_targets, all_masks = [], [], []
    for batch in val_loader:
        mel = batch["mel"].to(DEVICE)
        emotion = batch["emotion"].to(DEVICE)
        mask = batch["mask"].to(DEVICE)

        outputs = model(mel, active_experts=["emotion"])
        pred = outputs["emotion"]

        all_preds.append(pred.cpu())
        all_targets.append(emotion.cpu())
        all_masks.append(mask.cpu())

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

    print(f"\n{'='*50}")
    print(f"  {args.checkpoint}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
