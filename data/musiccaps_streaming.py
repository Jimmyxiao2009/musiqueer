"""
MusicCaps 流式加载器

Google MusicCaps 数据集: 5,521 条音乐片段 + 自由文本描述。
每项提供 YouTube ID + 起止时间。

流式方案: yt-dlp pipe 流式拉取音频 → mel 缓存。
文本描述可作为 CLAP-style 对比学习的辅助信号。

数据来源: https://huggingface.co/datasets/google/MusicCaps

用法:
    # 下载元数据
    python data/musiccaps_streaming.py --download-meta

    # 预处理 (流式拉取 → 缓存 mel)
    python data/musiccaps_streaming.py --preprocess --workers 2

    # 查看样本
    python data/musiccaps_streaming.py --show-samples
"""

import os
import sys
import io
import time
import json
import argparse
import csv
from typing import Optional, List, Dict

import torch
import requests
import pandas as pd

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

from data.youtube_streaming import (
    preprocess_youtube_dataset,
    YouTubeCacheDataset,
    stream_and_extract_mel,
)

# ── 常量 ──────────────────────────────────────────────────────────────
CACHE_DIR = os.path.join(_root, "data", "musiccaps_cache")
META_DIR = os.path.join(_root, "data", "musiccaps_meta")

# MusicCaps 元数据 URL (HuggingFace)
HF_DATASET_URL = "https://huggingface.co/datasets/google/MusicCaps/resolve/main/data"

# 官方 CSV 文件
CAPTIONS_URL = "https://storage.googleapis.com/musiccaps-public/musiccaps_public.csv"
# 备用: HF 镜像
HF_CAPTIONS_URL = (
    "https://huggingface.co/datasets/google/MusicCaps/resolve/main/data/musiccaps_public.csv"
)


# ── 元数据加载 ────────────────────────────────────────────────────────
def download_metadata():
    """下载 MusicCaps CSV (包含 YouTube ID + 起止时间 + 描述)"""
    os.makedirs(META_DIR, exist_ok=True)
    csv_path = os.path.join(META_DIR, "musiccaps_public.csv")

    if os.path.exists(csv_path):
        print(f"[MusicCaps] 已缓存: {csv_path}")
        return csv_path

    # 尝试主 URL
    for url, label in [(CAPTIONS_URL, "主站"), (HF_CAPTIONS_URL, "HF镜像")]:
        print(f"[MusicCaps] 从 {label} 下载...")
        try:
            resp = requests.get(url, stream=True, timeout=120)
            resp.raise_for_status()
            with open(csv_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"[MusicCaps] -> {csv_path} ({os.path.getsize(csv_path)} bytes)")
            return csv_path
        except Exception as e:
            print(f"[MusicCaps] {label} 下载失败: {e}")

    print("[MusicCaps] 所有下载源均失败")
    return None


def load_items(limit: int = 0) -> List[dict]:
    """加载 MusicCaps 条目"""
    csv_path = download_metadata()
    if not csv_path:
        return []

    items = []
    try:
        df = pd.read_csv(csv_path)
        print(f"[MusicCaps] CSV 列: {list(df.columns)}")

        for _, row in df.iterrows():
            # YouTube ID 列
            yt_col = None
            for c in df.columns:
                if "ytid" in c.lower() or "youtube" in c.lower() or "video" in c.lower():
                    yt_col = c
                    break
            if not yt_col:
                # 第一个看起来像 ID 的列
                for c in df.columns:
                    val = str(row[c])
                    if len(val) == 11 and val.isalnum():
                        yt_col = c
                        break

            if not yt_col:
                continue

            yid = str(row[yt_col]).strip()
            if not yid or len(yid) != 11:
                continue

            item = {"youtube_id": yid}

            # 起止时间
            for t_col in ["start_s", "start_time", "start"]:
                if t_col in df.columns:
                    item["start_s"] = float(row[t_col])
                    break

            # 描述文本
            for c_col in ["caption", "description", "text", "caption_en"]:
                if c_col in df.columns:
                    item["caption"] = str(row[c_col])
                    break

            # 其他元数据
            for c in ["genre", "instrument", "mood", "aspect_list", "audioset_positive_labels"]:
                if c in df.columns:
                    item[c] = str(row[c]) if pd.notna(row.get(c)) else ""

            items.append(item)
            if limit > 0 and len(items) >= limit:
                break

    except Exception as e:
        print(f"[MusicCaps] 加载失败: {e}")

    print(f"[MusicCaps] 加载了 {len(items)} 项")
    return items


# ── 预处理 ────────────────────────────────────────────────────────────
def preprocess(workers: int = 1, limit: int = 0):
    """下载元数据 + 流式处理所有条目"""
    items = load_items(limit=limit)
    if not items:
        print("[MusicCaps] 无数据可处理")
        return

    preprocess_youtube_dataset(
        items,
        id_field="youtube_id",
        cache_dir=CACHE_DIR,
        label_fn=None,  # 无结构化标签
        extra_fields=["caption", "start_s", "genre", "instrument", "mood"],
        num_workers=workers,
        limit=0,  # 已在 load_items 中处理 limit
    )


# ── Dataset ───────────────────────────────────────────────────────────
class MusicCapsDataset(YouTubeCacheDataset):
    """MusicCaps 数据集 (从 mel 缓存读取)"""

    def __init__(self, cache_dir: str = None, split: str = None):
        super().__init__(cache_dir or CACHE_DIR, split=split)


def create_dataloaders(batch_size: int = 16, split: str = None, num_workers: int = 0):
    """便捷接口"""
    dataset = MusicCapsDataset(split=split)
    if len(dataset) == 0:
        return None
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=(split != "test"),
        num_workers=num_workers, pin_memory=True,
    )


def show_samples(n: int = 3):
    """显示几个样本的内容"""
    items = load_items(limit=n)
    for i, item in enumerate(items):
        print(f"\n--- 样本 {i+1} ---")
        print(f"  YouTube ID: {item.get('youtube_id')}")
        print(f"  Caption: {item.get('caption', 'N/A')[:200]}")
        print(f"  Start: {item.get('start_s', 'N/A')}s")
        for k in ["genre", "instrument", "mood"]:
            if k in item and item[k]:
                print(f"  {k}: {item[k][:100]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MusicCaps 流式预处理")
    parser.add_argument("--download-meta", action="store_true", help="下载元数据")
    parser.add_argument("--preprocess", action="store_true", help="预处理所有条目")
    parser.add_argument("--workers", type=int, default=1, help="并行数")
    parser.add_argument("--limit", type=int, default=0, help="限制处理数")
    parser.add_argument("--show-samples", action="store_true", help="显示样本")
    args = parser.parse_args()

    if args.download_meta:
        download_metadata()

    if args.preprocess:
        preprocess(workers=args.workers, limit=args.limit)

    if args.show_samples:
        show_samples()

    if not any([args.download_meta, args.preprocess, args.show_samples]):
        show_samples()
