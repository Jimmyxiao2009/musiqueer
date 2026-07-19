"""
YouTube 音频流式加载器 — 通用模块

通过 yt-dlp pipe 音频到 stdout → librosa 实时解码，零落盘。
适用于所有提供 YouTube ID 的数据集 (Music4All, MusicCaps, AudioSet 等)。

用法:
    # 直接流式单曲
    audio = stream_youtube_audio("dQw4w9WgXcQ", target_sr=22050, duration=30)

    # 预处理并缓存 (推荐)
    python data/youtube_streaming.py --csv data/music4all_meta/tracks.csv --id_col youtube_id
"""

import os
import sys
import json
import time
import io
import subprocess
import tempfile
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Callable, List

import torch
import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

from utils.audio import load_audio, mel_spectrogram, aggregate_frames

# ── 常量 ──────────────────────────────────────────────────────────────
SAMPLE_RATE = 22050
N_MELS = 128
FRAME_AGG = 21
TARGET_FRAMES = 61           # 30s (对齐 MTG-Jamendo)
MAX_DURATION = 30.0

YT_DLP_CMD = "yt-dlp"


def stream_youtube_audio(youtube_id: str, target_sr: int = SAMPLE_RATE,
                         duration: float = MAX_DURATION,
                         yt_dlp_cmd: str = YT_DLP_CMD,
                         timeout: int = 60) -> Optional[np.ndarray]:
    """
    通过 yt-dlp pipe 流式拉取 YouTube 音频，返回 numpy 波形数组。

    原理:
        yt-dlp -x --audio-format wav --output - \
            --download-sections "*0-{duration}" \
            https://www.youtube.com/watch?v={id}

    零文件落盘: 音频数据通过 stdout pipe 直接送入 librosa 解码。

    Args:
        youtube_id: YouTube 视频 ID
        target_sr: 目标采样率
        duration: 截取前 N 秒
        yt_dlp_cmd: yt-dlp 命令路径
        timeout: 进程超时秒数

    Returns:
        numpy array [T] (mono), 失败时返回 None
    """
    url = f"https://www.youtube.com/watch?v={youtube_id}"
    cmd = [
        yt_dlp_cmd, "-x", "--audio-format", "wav",
        "--output", "-",                    # stdout
        "--download-sections", f"*0-{duration}",
        "--force-keyframes-at-cuts",
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        url,
    ]

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )

        # 从 pipe 读取 wav → librosa 解码
        import librosa
        audio, _ = librosa.load(
            io.BytesIO(proc.stdout.read()),
            sr=target_sr, mono=True,
        )
        proc.stdout.close()
        proc.wait(timeout=timeout)
        return audio

    except Exception as e:
        # 静默失败，让调用方处理
        return None


def stream_and_extract_mel(youtube_id: str, target_sr: int = SAMPLE_RATE,
                           duration: float = MAX_DURATION) -> Optional[torch.Tensor]:
    """
    流式拉取 YouTube 音频 → 提取 mel 频谱 → 返回 Tensor。

    Returns:
        mel: [1, n_mels, T] 或 None (失败)
    """
    audio = stream_youtube_audio(youtube_id, target_sr, duration)
    if audio is None:
        return None

    # 转 torch tensor [1, T]
    wav = torch.from_numpy(audio).unsqueeze(0).float()

    # 截取/填充到 max_duration
    max_samples = int(target_sr * duration)
    if wav.shape[-1] > max_samples:
        wav = wav[..., :max_samples]
    elif wav.shape[-1] < max_samples:
        wav = torch.nn.functional.pad(wav, (0, max_samples - wav.shape[-1]))

    mel = mel_spectrogram(wav, target_sr, n_mels=N_MELS)
    mel = aggregate_frames(mel, FRAME_AGG)

    # 对齐到 TARGET_FRAMES
    T = mel.shape[-1]
    if T > TARGET_FRAMES:
        mel = mel[..., :TARGET_FRAMES]
    elif T < TARGET_FRAMES:
        mel = torch.nn.functional.pad(mel, (0, TARGET_FRAMES - T))

    return mel


def preprocess_youtube_dataset(
    items: List[dict],
    id_field: str = "youtube_id",
    cache_dir: str = "yt_cache",
    label_fn: Optional[Callable] = None,
    extra_fields: Optional[List[str]] = None,
    num_workers: int = 1,
    limit: int = 0,
):
    """
    通用 YouTube ID 数据集预处理。

    Args:
        items: 列表，每项至少包含 id_field 指定的字段
        id_field: 存放 YouTube ID 的字段名
        cache_dir: 缓存目录
        label_fn: items[i] → torch.Tensor, 返回标签张量
        extra_fields: 需要保存到 .pt 缓存中的额外字段名
        num_workers: 并行流式下载数
        limit: 限制处理条目数 (调试用)
    """
    os.makedirs(cache_dir, exist_ok=True)
    extra_fields = extra_fields or []

    if limit > 0:
        items = items[:limit]

    print(f"[YouTube] 开始预处理 {len(items)} 条 (workers={num_workers})")

    n_ok, n_skip, n_fail = 0, 0, 0
    t0 = time.time()

    if num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = []
            for item in items:
                yid = item.get(id_field)
                if not yid:
                    n_fail += 1
                    continue

                cache_path = os.path.join(cache_dir, f"{yid}.pt")
                if os.path.exists(cache_path):
                    n_skip += 1
                    continue

                futures.append(pool.submit(
                    _process_one, yid, item, cache_dir, label_fn, extra_fields
                ))

            for f in as_completed(futures):
                ok, yid = f.result()
                if ok:
                    n_ok += 1
                else:
                    n_fail += 1

                if (n_ok + n_fail + n_skip) % 50 == 0:
                    elapsed = time.time() - t0
                    print(f"  进度: {n_ok} ok / {n_fail} fail / {n_skip} skip, {elapsed:.0f}s")
    else:
        for item in items:
            yid = item.get(id_field)
            if not yid:
                n_fail += 1
                continue

            cache_path = os.path.join(cache_dir, f"{yid}.pt")
            if os.path.exists(cache_path):
                n_skip += 1
                continue

            ok, _ = _process_one(yid, item, cache_dir, label_fn, extra_fields)
            if ok:
                n_ok += 1
            else:
                n_fail += 1

            if (n_ok + n_fail) % 100 == 0:
                elapsed = time.time() - t0
                print(f"  进度: {n_ok} ok / {n_fail} fail / {n_skip} skip, {elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"[YouTube] 完成: {n_ok} ok, {n_skip} skip, {n_fail} fail, {elapsed:.0f}s")


def _process_one(yid: str, item: dict, cache_dir: str,
                 label_fn: Optional[Callable], extra_fields: List[str]):
    """处理单条 YouTube ID（供线程池调用）"""
    try:
        mel = stream_and_extract_mel(yid)
        if mel is None:
            return False, yid

        save_dict = {"mel": mel, "youtube_id": yid}

        if label_fn:
            save_dict["label"] = label_fn(item)

        for field in extra_fields:
            if field in item:
                save_dict[field] = item[field]

        torch.save(save_dict, os.path.join(cache_dir, f"{yid}.pt"))
        return True, yid

    except Exception:
        return False, yid


class YouTubeCacheDataset(torch.utils.data.Dataset):
    """
    通用 YouTube 缓存数据集 — 从 .pt 缓存读取。
    缓存由 preprocess_youtube_dataset() 生成。
    """

    def __init__(self, cache_dir: str, split: str = None,
                 split_field: str = "split"):
        self.cache_dir = cache_dir
        self.split = split
        self.split_field = split_field

        all_files = sorted([f for f in os.listdir(cache_dir) if f.endswith(".pt")])

        if split:
            self.files = []
            for f in all_files:
                try:
                    data = torch.load(os.path.join(cache_dir, f),
                                      map_location="cpu", weights_only=True)
                    if data.get(split_field) == split:
                        self.files.append(f)
                except Exception:
                    pass
            print(f"[YouTubeCacheDataset] {len(self.files)}/{len(all_files)} (split={split})")
        else:
            self.files = all_files
            print(f"[YouTubeCacheDataset] {len(self.files)} 样本 (no split)")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(
            os.path.join(self.cache_dir, self.files[idx]),
            map_location="cpu", weights_only=True,
        )
        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YouTube 音频流式预处理")
    parser.add_argument("--csv", required=True, help="元数据 CSV 路径")
    parser.add_argument("--id_col", default="youtube_id", help="YouTube ID 列名")
    parser.add_argument("--label_col", default=None, help="标签列名 (可选)")
    parser.add_argument("--cache_dir", default=None, help="缓存目录")
    parser.add_argument("--workers", type=int, default=1, help="并行数")
    parser.add_argument("--limit", type=int, default=0, help="限制处理数")
    args = parser.parse_args()

    import pandas as pd
    df = pd.read_csv(args.csv)
    items = df.to_dict("records")

    cache_dir = args.cache_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", f"yt_cache_{os.path.splitext(os.path.basename(args.csv))[0]}"
    )

    label_fn = None
    if args.label_col:
        def make_label_fn(col):
            # 自动检测: 分类 (单值) vs 多标签 (逗号分隔)
            def fn(item):
                val = item[col]
                if isinstance(val, str) and "," in val:
                    parts = [p.strip() for p in val.split(",") if p.strip()]
                    label = torch.zeros(len(parts))
                    for i, p in enumerate(sorted(set(parts))):
                        label[i] = 1.0
                    return label
                else:
                    return torch.tensor(float(val) if val else 0.0)
            return fn
        label_fn = make_label_fn(args.label_col)

    extra_fields = [c for c in df.columns if c not in (args.id_col, args.label_col)]

    preprocess_youtube_dataset(
        items, id_field=args.id_col, cache_dir=cache_dir,
        label_fn=label_fn, extra_fields=extra_fields,
        num_workers=args.workers, limit=args.limit,
    )
