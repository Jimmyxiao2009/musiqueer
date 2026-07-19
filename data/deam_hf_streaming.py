"""
DEAM HuggingFace 流式加载器

使用 HuggingFace `datasets` 库的 `streaming=True` 模式加载 DEAM 数据集，
避免一次性下载全部音频到本地。

数据源: dblzdw/DEAM (HuggingFace)

备用: 如果 HF 网络不可用，回退到本地 DEAM 数据。

用法:
    python data/deam_hf_streaming.py --show-samples
    python data/deam_hf_streaming.py --preprocess --limit 500
"""

import os
import sys
import time
import json
import tempfile
import argparse
from typing import Optional, List, Dict

import torch
import numpy as np
from pathlib import Path

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

from utils.audio import load_audio, mel_spectrogram, aggregate_frames

# ── 常量 ──────────────────────────────────────────────────────────────
SAMPLE_RATE = 22050
N_MELS = 128
FRAME_AGG = 21
TARGET_FRAMES = 30           # DEAM 用 30 帧 (15s)
MAX_DURATION = 15.0

CACHE_DIR = os.path.join(_root, "data", "deam_hf_cache")

HF_DATASET_NAME = "dblzdw/DEAM"

# 备用: 如果 HF streaming 不可用，尝试从 HF Hub 下载原始文件
HF_FILES = {
    "val_annotations": (
        "https://huggingface.co/datasets/dblzdw/DEAM/resolve/main/data/val_annotations.csv"
    ),
    "train_annotations": (
        "https://huggingface.co/datasets/dblzdw/DEAM/resolve/main/data/train_annotations.csv"
    ),
}


# ── HF 流式加载 ───────────────────────────────────────────────────────
def try_hf_streaming() -> bool:
    """测试 HuggingFace datasets streaming 是否可用"""
    try:
        from datasets import load_dataset
        # 只加载一个样本测试连接
        ds = load_dataset(HF_DATASET_NAME, split="train", streaming=True)
        for i, sample in enumerate(ds):
            if i >= 1:
                break
        print(f"[DEAM HF] streaming 连接成功: {HF_DATASET_NAME}")
        return True
    except Exception as e:
        print(f"[DEAM HF] streaming 不可用: {e}")
        return False


def load_hf_streaming(split: str = "train", limit: int = 0):
    """
    用 HuggingFace datasets streaming 模式加载 DEAM。
    自动 fallback: streaming → 本地缓存 → 本地 DEAM 数据
    """
    try:
        from datasets import load_dataset

        print(f"[DEAM HF] 加载 {split} split (streaming=True)...")
        ds = load_dataset(HF_DATASET_NAME, split=split, streaming=True)

        samples = []
        for i, sample in enumerate(ds):
            if limit > 0 and i >= limit:
                break
            samples.append(sample)

        print(f"[DEAM HF] 加载了 {len(samples)} 个样本")
        return samples

    except Exception as e:
        print(f"[DEAM HF] streaming 加载失败: {e}")
        print("[DEAM HF] 尝试从本地 DEAM 数据回退...")
        return _fallback_to_local(split, limit)


def _fallback_to_local(split: str, limit: int) -> list:
    """回退到 data/deam_dataset.py 的本地数据"""
    try:
        from data.deam_dataset import create_dataloaders

        loader = create_dataloaders(batch_size=1, split=split)
        if loader is None:
            return []

        samples = []
        for i, batch in enumerate(loader):
            if limit > 0 and i >= limit:
                break
            samples.append({
                "mel": batch["mel"][0],
                "emotion": batch["emotion"][0],
                "mask": batch["mask"][0],
            })

        print(f"[DEAM HF 回退] 从本地加载了 {len(samples)} 个样本")
        return samples

    except Exception as e:
        print(f"[DEAM HF 回退] 也失败: {e}")
        return []


# ── 预处理: HF → mel 缓存 ─────────────────────────────────────────────
def preprocess(split: str = "train", limit: int = 0):
    """从 HF streaming 加载 DEAM 样本，提取 mel 并缓存"""
    os.makedirs(CACHE_DIR, exist_ok=True)

    samples = load_hf_streaming(split, limit)
    if not samples:
        print("[DEAM HF] 无样本可处理")
        return

    n_ok, n_skip = 0, 0
    t0 = time.time()

    for i, sample in enumerate(samples):
        # 生成缓存键
        song_id = sample.get("song_id", i)
        segment_id = sample.get("segment_id", 0)
        cache_key = f"deam_{song_id}_{segment_id}"
        cache_path = os.path.join(CACHE_DIR, f"{cache_key}.pt")

        if os.path.exists(cache_path):
            n_skip += 1
            continue

        try:
            # 从 sample 提取音频 → mel
            audio_array = None
            # HF datasets 的音频列通常是 {"path": ..., "array": ...}
            if "audio" in sample:
                audio_data = sample["audio"]
                if isinstance(audio_data, dict):
                    audio_array = audio_data.get("array")
                    sample_rate = audio_data.get("sampling_rate", SAMPLE_RATE)
                    wav = torch.from_numpy(audio_array).float()
                    if wav.dim() == 1:
                        wav = wav.unsqueeze(0)
                    # 重采样到 target_sr
                    if sample_rate != SAMPLE_RATE:
                        import torchaudio
                        resampler = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)
                        wav = resampler(wav)
                else:
                    # 可能是文件路径
                    if isinstance(audio_data, str) and os.path.exists(audio_data):
                        wav, sr = load_audio(audio_data, SAMPLE_RATE, MAX_DURATION)
                    else:
                        continue

            elif "mel" in sample:
                # 可能已有预计算 mel
                save_dict = {k: v for k, v in sample.items() if isinstance(v, torch.Tensor)}
                torch.save(save_dict, cache_path)
                n_ok += 1
                continue
            else:
                # 只有标注没有音频
                continue

            mel = mel_spectrogram(wav, SAMPLE_RATE, n_mels=N_MELS)
            mel = aggregate_frames(mel, FRAME_AGG)

            T = mel.shape[-1]
            if T > TARGET_FRAMES:
                mel = mel[..., :TARGET_FRAMES]
            elif T < TARGET_FRAMES:
                mel = torch.nn.functional.pad(mel, (0, TARGET_FRAMES - T))

            # 保存
            save_dict = {"mel": mel}

            # 复制有用的标注
            if "emotion" in sample:
                save_dict["emotion"] = torch.tensor(sample["emotion"]) if not isinstance(
                    sample["emotion"], torch.Tensor) else sample["emotion"]
            if "valence" in sample:
                save_dict["valence"] = torch.tensor(sample["valence"])
            if "arousal" in sample:
                save_dict["arousal"] = torch.tensor(sample["arousal"])
            if "mask" in sample:
                save_dict["mask"] = torch.tensor(sample["mask"]) if not isinstance(
                    sample["mask"], torch.Tensor) else sample["mask"]

            torch.save(save_dict, cache_path)
            n_ok += 1

        except Exception as e:
            continue

        if (n_ok + n_skip) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{split}] {n_ok} ok / {n_skip} skip, {elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"[DEAM HF {split}] 完成: {n_ok} ok, {n_skip} skip, {elapsed:.0f}s")


# ── Dataset ───────────────────────────────────────────────────────────
class DEAMHFDataset(torch.utils.data.Dataset):
    """从 HF streaming 缓存读取 DEAM 数据"""

    def __init__(self, cache_dir: str = None, split: str = None):
        self.cache_dir = cache_dir or CACHE_DIR
        self.split = split
        self.files = sorted([f for f in os.listdir(self.cache_dir) if f.endswith(".pt")])
        print(f"[DEAMHFDataset] {len(self.files)} 样本 (split={split})")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(
            os.path.join(self.cache_dir, self.files[idx]),
            map_location="cpu", weights_only=True,
        )
        return data


def create_dataloaders(batch_size: int = 16, split: str = None, num_workers: int = 0):
    """便捷接口"""
    dataset = DEAMHFDataset(split=split)
    if len(dataset) == 0:
        return None
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=(split != "test"),
        num_workers=num_workers, pin_memory=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DEAM HuggingFace 流式预处理")
    parser.add_argument("--test-connection", action="store_true", help="测试 HF 连接")
    parser.add_argument("--preprocess", action="store_true", help="预处理数据")
    parser.add_argument("--limit", type=int, default=0, help="限制处理数")
    args = parser.parse_args()

    if args.test_connection:
        ok = try_hf_streaming()
        print(f"HF streaming: {'可用' if ok else '不可用'}")

    if args.preprocess:
        for split in ["train", "validation"]:
            preprocess(split=split, limit=args.limit)

    if not args.test_connection and not args.preprocess:
        ok = try_hf_streaming()
        print(f"HF streaming: {'✅ 可用' if ok else '❌ 不可用'}")
