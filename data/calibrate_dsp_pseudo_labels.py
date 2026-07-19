"""
预跑一遍DEAM训练集样本，估计Tension/Complexity DSP伪标签(见dsp_pseudo_labels.py)
子特征的全局均值/方差，供训练时做z-score归一化。跑一次即可，结果落盘到
data/dsp_pseudo_label_stats.json。

用法:
    python data/calibrate_dsp_pseudo_labels.py [--n_samples 400]
"""

import os
import sys
import json
import argparse

import numpy as np
import librosa

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

from data.deam_dataset import create_dataloaders, SAMPLE_RATE, SEGMENT_SEC
from data.dsp_pseudo_labels import (
    extract_raw_features, RAW_FEATURE_NAMES, STATS_PATH,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=400)
    parser.add_argument("--deam_root", type=str, default="D:/Musiclearner/data/deam")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("[数据] 加载DEAM train segment列表(只用来抽样，不跑模型)...")
    _, _, train_dataset, _ = create_dataloaders(args.deam_root, batch_size=4, num_workers=0)
    segments = train_dataset.segments
    audio_dir = train_dataset.audio_dir

    rng = np.random.RandomState(args.seed)
    idx = rng.choice(len(segments), size=min(args.n_samples, len(segments)), replace=False)

    raw_accum = {name: [] for name in RAW_FEATURE_NAMES}
    raw_per_sample = []  # 缓存每个片段的raw dict，避免二次加载音频重复算STFT

    print(f"[采样] {len(idx)}个片段，逐个提取DSP特征...")
    for n, i in enumerate(idx):
        song_id, start_sec = segments[i]
        mp3_path = os.path.join(audio_dir, f"{song_id}.mp3")
        try:
            y, _ = librosa.load(mp3_path, sr=SAMPLE_RATE, mono=True,
                                offset=start_sec, duration=SEGMENT_SEC)
        except Exception as e:
            print(f"  [跳过] song={song_id} t={start_sec}: {e}")
            continue
        if len(y) < SAMPLE_RATE:  # 太短跳过，避免STFT退化
            continue
        raw = extract_raw_features(y, SAMPLE_RATE)
        raw_per_sample.append(raw)
        for name in RAW_FEATURE_NAMES:
            raw_accum[name].append(raw[name])
        if (n + 1) % 50 == 0:
            print(f"  {n + 1}/{len(idx)}")

    stats = {}
    for name in RAW_FEATURE_NAMES:
        vals = np.concatenate(raw_accum[name])
        stats[name] = {"mean": float(vals.mean()), "std": float(vals.std())}
        print(f"  {name:24s} mean={stats[name]['mean']:+.4f}  std={stats[name]['std']:.4f}")

    # 用刚算出的子特征stats，计算composite(加权组合后)的全局均值/方差，供二次z-score
    print("[组合] 计算tension/complexity组合分数的全局统计...")
    from data.dsp_pseudo_labels import compute_composites
    tension_all, complexity_all = [], []
    for raw in raw_per_sample:
        t, c = compute_composites(raw, stats)
        tension_all.append(t)
        complexity_all.append(c)

    tension_all = np.concatenate(tension_all)
    complexity_all = np.concatenate(complexity_all)
    stats["tension_composite"] = {"mean": float(tension_all.mean()),
                                  "std": float(tension_all.std())}
    stats["complexity_composite"] = {"mean": float(complexity_all.mean()),
                                     "std": float(complexity_all.std())}
    print(f"  tension_composite    mean={stats['tension_composite']['mean']:+.4f} "
          f"std={stats['tension_composite']['std']:.4f}")
    print(f"  complexity_composite mean={stats['complexity_composite']['mean']:+.4f} "
          f"std={stats['complexity_composite']['std']:.4f}")

    stats["_meta"] = {"n_samples": int(len(tension_all) > 0 and len(idx)),
                      "seed": args.seed}

    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n[完成] 统计已保存 → {STATS_PATH}")


if __name__ == "__main__":
    main()
