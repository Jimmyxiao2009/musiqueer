"""
Music4All-Onion 流式加载器

113K 首音乐曲目，含 YouTube ID + 流派多标签。
流式方案: 通过 yt-dlp pipe 从 YouTube 流式拉取音频，缓存 mel 特征。
标签: 从元数据 CSV 读取流派信息。

数据来源: https://zenodo.org/records/17400116 (Music4All A+A)

用法:
    # 先下载元数据
    python data/music4all_streaming.py --download-meta

    # 预处理 (流式拉取音频 → 缓存 mel)
    python data/music4all_streaming.py --preprocess --limit 1000

    # 一次完成
    python data/music4all_streaming.py --download-meta --preprocess --workers 2
"""

import os
import sys
import io
import time
import argparse
import zipfile
import requests
import json
from pathlib import Path
from typing import Optional, List, Dict

import torch
import numpy as np
import pandas as pd

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

from data.youtube_streaming import (
    preprocess_youtube_dataset,
    YouTubeCacheDataset,
    stream_and_extract_mel,
)

# ── 常量 ──────────────────────────────────────────────────────────────
CACHE_DIR = os.path.join(_root, "data", "music4all_cache")
META_DIR = os.path.join(_root, "data", "music4all_meta")

# Music4All A+A Zenodo
ZENODO_URL = "https://zenodo.org/records/17400116"
# 实际数据文件
META_URLS = {
    "tracks": "https://zenodo.org/records/17400116/files/tracks.csv?download=1",
    "genres": "https://zenodo.org/records/17400116/files/genres.csv?download=1",
}

# 可用的 YouTube ID 列表 (从 Music4All 原始仓库)
# 备用: 从 HuggingFace 社区镜像加载
HF_META_URL = (
    "https://huggingface.co/datasets/seungheondoh/music4all-onion/raw/main/metadata.csv"
)


# ── 元数据加载 ────────────────────────────────────────────────────────
def _download_metadata():
    """下载 Music4All 元数据"""
    os.makedirs(META_DIR, exist_ok=True)

    # 尝试从 Zenodo 下载
    for name, url in META_URLS.items():
        path = os.path.join(META_DIR, f"{name}.csv")
        if os.path.exists(path):
            print(f"[Music4All] 已缓存: {path}")
            continue

        print(f"[Music4All] 下载 {name}.csv 从 Zenodo...")
        try:
            resp = requests.get(url, stream=True, timeout=120)
            resp.raise_for_status()
            with open(path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"[Music4All] -> {path} ({os.path.getsize(path)} bytes)")
        except Exception as e:
            print(f"[Music4All] Zenodo 下载失败: {e}")
            print("[Music4All] 尝试从 HuggingFace 镜像下载...")
            _download_from_hf()

    # 尝试获取 YouTube ID 列表
    yt_path = os.path.join(META_DIR, "tracks.csv")
    yt_ids_path = os.path.join(META_DIR, "youtube_ids.txt")
    if not os.path.exists(yt_ids_path):
        _extract_youtube_ids()


def _download_from_hf():
    """从 HuggingFace 社区镜像下载元数据"""
    try:
        resp = requests.get(HF_META_URL, stream=True, timeout=120)
        resp.raise_for_status()
        path = os.path.join(META_DIR, "metadata.csv")
        with open(path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"[Music4All] HF 元数据 -> {path}")
    except Exception as e:
        print(f"[Music4All] HF 也失败: {e}")


def _extract_youtube_ids():
    """从 tracks.csv 中提取 YouTube ID"""
    tracks_path = os.path.join(META_DIR, "tracks.csv")
    if not os.path.exists(tracks_path):
        print("[Music4All] tracks.csv 不存在，跳过 YouTube ID 提取")
        return

    try:
        df = pd.read_csv(tracks_path)
        # 自动检测 YouTube ID 列
        yt_candidates = [c for c in df.columns
                         if "youtube" in c.lower() or "yt" in c.lower() or "url" in c.lower()]
        if yt_candidates:
            yt_col = yt_candidates[0]
            ids = df[yt_col].dropna().unique().tolist()
        else:
            # 尝试常见列名
            for col in ["url", "link", "id", "video_id", "youtube_url"]:
                if col in df.columns:
                    ids = df[col].dropna().unique().tolist()
                    break
            else:
                print("[Music4All] 未找到 YouTube ID 列")
                return

        yt_path = os.path.join(META_DIR, "youtube_ids.txt")
        with open(yt_path, "w") as f:
            for yid in ids:
                # 从 URL 中提取 ID
                yid_str = str(yid).strip()
                if "youtube.com/watch?v=" in yid_str:
                    yid_str = yid_str.split("v=")[-1].split("&")[0]
                elif "youtu.be/" in yid_str:
                    yid_str = yid_str.split("youtu.be/")[-1].split("?")[0]
                if yid_str and len(yid_str) == 11:
                    f.write(yid_str + "\n")
        print(f"[Music4All] 提取了 {len(ids)} 个 YouTube ID -> {yt_path}")

    except Exception as e:
        print(f"[Music4All] YouTube ID 提取失败: {e}")


def load_items(limit: int = 0) -> List[dict]:
    """加载 Music4All 曲目列表，每项包含 youtube_id 和 (可选) 标签"""
    # 尝试加载缓存的元数据
    meta_paths = [
        os.path.join(META_DIR, "tracks.csv"),
        os.path.join(META_DIR, "metadata.csv"),
    ]

    items = []
    seen_ids = set()

    for mp in meta_paths:
        if not os.path.exists(mp):
            continue
        try:
            df = pd.read_csv(mp)
            # 找 YouTube ID 列
            yt_col = None
            for c in df.columns:
                if "youtube" in c.lower() or "yt_id" in c.lower() or "url" in c.lower():
                    yt_col = c
                    break
            if not yt_col:
                continue

            for _, row in df.iterrows():
                yid = str(row[yt_col]).strip()
                if not yid:
                    continue
                # 从 URL 提取 ID
                if "youtube.com/watch?v=" in yid:
                    yid = yid.split("v=")[-1].split("&")[0]
                elif "youtu.be/" in yid:
                    yid = yid.split("youtu.be/")[-1].split("?")[0]
                if len(yid) != 11:
                    continue
                if yid in seen_ids:
                    continue
                seen_ids.add(yid)

                item = {"youtube_id": yid}
                # 尝试读取标签
                genre_cols = [c for c in df.columns
                              if "genre" in c.lower() or "tag" in c.lower() or "label" in c.lower()]
                if genre_cols:
                    item["genre"] = str(row[genre_cols[0]])

                items.append(item)
                if limit > 0 and len(items) >= limit:
                    break

            if items:
                print(f"[Music4All] 从 {mp} 加载了 {len(items)} 项")
        except Exception as e:
            print(f"[Music4All] 加载 {mp} 失败: {e}")

    # 如果元数据不存在，使用预置的采样 ID 做功能演示
    if not items:
        print("[Music4All] 元数据未找到，使用预设演示 ID")
        # 一些公共领域音乐 YouTube ID 做测试
        demo_ids = [
            "dQw4w9WgXcQ",  # Rick Astley - Never Gonna Give You Up
        ]
        items = [{"youtube_id": yid} for yid in demo_ids]

    return items


# ── 预处理 ────────────────────────────────────────────────────────────
def preprocess(workers: int = 1, limit: int = 0):
    """下载元数据 + 流式处理所有曲目"""
    _download_metadata()
    items = load_items(limit=limit)

    def label_fn(item):
        genre = item.get("genre", "")
        if genre:
            return torch.tensor(hash(genre) % 1000)  # placeholder
        return None

    preprocess_youtube_dataset(
        items,
        id_field="youtube_id",
        cache_dir=CACHE_DIR,
        label_fn=label_fn if any("genre" in i for i in items) else None,
        extra_fields=[],
        num_workers=workers,
        limit=limit if limit > 0 else 0,
    )


# ── Dataset ───────────────────────────────────────────────────────────
class Music4AllDataset(YouTubeCacheDataset):
    """Music4All 数据集 (从 mel 缓存读取)"""

    def __init__(self, cache_dir: str = None, split: str = None):
        super().__init__(cache_dir or CACHE_DIR, split=split)


def create_dataloaders(batch_size: int = 16, split: str = None, num_workers: int = 0):
    """便捷接口"""
    dataset = Music4AllDataset(split=split)
    if len(dataset) == 0:
        return None
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=(split != "test"),
        num_workers=num_workers, pin_memory=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Music4All 流式预处理")
    parser.add_argument("--download-meta", action="store_true", help="下载元数据")
    parser.add_argument("--preprocess", action="store_true", help="预处理所有曲目")
    parser.add_argument("--workers", type=int, default=1, help="并行数")
    parser.add_argument("--limit", type=int, default=0, help="限制处理数")
    args = parser.parse_args()

    if args.download_meta:
        _download_metadata()

    if args.preprocess:
        preprocess(workers=args.workers, limit=args.limit)

    if not args.download_meta and not args.preprocess:
        items = load_items(limit=5)
        print(f"[Music4All] 加载了 {len(items)} 项 (演示)")
        for item in items[:3]:
            print(f"  {item}")
