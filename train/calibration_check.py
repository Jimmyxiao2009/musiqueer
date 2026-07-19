"""
诊断实验：Valence 预测的 R² 差是不是"尺度/偏差没校准好"，而不是"完全没学到"。

背景: joint_best_corr_model.pth 上 V_mean_R2=0.036 但 V_mean_PearsonR=0.338
(docs/PROGRESS.md "Step2 收尾")。R²同时惩罚尺度/偏差失配，PearsonR只看
相关方向。如果给预测值做个简单线性重标定(pred' = a*pred+b)就能让R²明显
改善，说明模型确实学到了valence的相对走向，只是输出没校准；如果校准后
R²还是很差，说明问题更根本，不是尺度问题。

方法论: 按歌曲切成两半，一半fit校准系数(a,b)，另一半算校准后R²——不能
在同一份数据上fit再测，那样R²数学上必然等于Pearson r²，是循环论证。
"""

import os
import sys
import numpy as np
import torch

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

from model.music_understand import MusicUnderstand2
from data.deam_dataset import create_dataloaders
import yaml

CONFIG_PATH = os.path.join(_root, "config", "default.yaml")
with open(CONFIG_PATH, encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def r2_score(pred, true):
    ss_res = ((true - pred) ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum()
    return 1 - ss_res / (ss_tot + 1e-8)


@torch.no_grad()
def main(checkpoint_path):
    model_config = CONFIG["model"]
    backbone_cfg = {k: v for k, v in model_config["backbone"].items() if k != "type"}
    emotion_cfg = {k: v for k, v in model_config["experts"]["emotion"].items()
                   if k not in ("loss_weight",)}
    mood_cfg = {"num_tags": 59}

    model = MusicUnderstand2(
        backbone_config=backbone_cfg, emotion_config=emotion_cfg, mood_config=mood_cfg,
    ).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"加载 {checkpoint_path} (epoch={ckpt.get('epoch')})")

    _, val_loader, _, _ = create_dataloaders(
        root_dir=CONFIG["paths"]["deam_root"], batch_size=16, num_workers=0,
    )

    preds, trues, song_ids = [], [], []
    for batch in val_loader:
        mel = batch["mel"].to(DEVICE)
        out = model(mel, active_experts=["emotion"])
        pred_v = out["emotion"][:, :, 0].mean(dim=1).cpu().numpy()  # 每段的V_mean均值
        true_v = batch["emotion"][:, :, 0].mean(dim=1).numpy()
        preds.append(pred_v)
        trues.append(true_v)
        song_ids.extend(batch["song_id"].tolist() if hasattr(batch["song_id"], "tolist")
                        else list(batch["song_id"]))

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    song_ids = np.array(song_ids)

    unique_songs = sorted(set(song_ids.tolist()))
    rng = np.random.RandomState(42)
    shuffled = unique_songs[:]
    rng.shuffle(shuffled)
    half = len(shuffled) // 2
    fit_songs = set(shuffled[:half])
    eval_songs = set(shuffled[half:])

    fit_mask = np.array([sid in fit_songs for sid in song_ids])
    eval_mask = np.array([sid in eval_songs for sid in song_ids])

    pred_fit, true_fit = preds[fit_mask], trues[fit_mask]
    pred_eval, true_eval = preds[eval_mask], trues[eval_mask]

    print(f"\n段数: fit={len(pred_fit)} eval={len(pred_eval)} "
          f"(按{len(fit_songs)}/{len(eval_songs)}首歌切分，不跨集)")

    # 在fit集上拟合线性校准 true ≈ a*pred + b
    a, b = np.polyfit(pred_fit, true_fit, deg=1)
    print(f"\n拟合的校准系数: a={a:.4f}, b={b:.4f}")

    pred_eval_calibrated = a * pred_eval + b

    raw_r2 = r2_score(torch.tensor(pred_eval), torch.tensor(true_eval)).item()
    calibrated_r2 = r2_score(torch.tensor(pred_eval_calibrated), torch.tensor(true_eval)).item()

    pred_std, true_std = pred_eval.std(), true_eval.std()
    pred_mean, true_mean = pred_eval.mean(), true_eval.mean()

    print(f"\n=== 结果 (仅在eval集上评估，未参与fit) ===")
    print(f"原始预测: mean={pred_mean:.4f} std={pred_std:.4f}")
    print(f"真实值:   mean={true_mean:.4f} std={true_std:.4f}")
    print(f"\n未校准 R²: {raw_r2:.4f}")
    print(f"校准后   R²: {calibrated_r2:.4f}")
    print(f"\n改善量: {calibrated_r2 - raw_r2:+.4f}")

    if calibrated_r2 - raw_r2 > 0.05:
        print("\n[结论] 校准带来明显改善——支持'尺度/偏差没校准好'的猜测。")
    elif calibrated_r2 - raw_r2 > 0.01:
        print("\n[结论] 校准带来小幅改善，但不足以说明这是主要问题。")
    else:
        print("\n[结论] 校准几乎没有帮助——问题不是尺度/偏差，模型确实没学到"
              "足够强的valence信号，即便相关系数看起来还行。")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/joint_best_corr_model.pth")
    args = parser.parse_args()
    main(args.checkpoint)
