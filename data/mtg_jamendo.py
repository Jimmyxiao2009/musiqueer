"""
MTG-Jamendo mood/theme 子集 —— 流式预处理管线

背景: DEAM只有1780首歌，emotion_head的Valence维度学不起来（R²≈0，语义信息
不够）。MTG-Jamendo autotagging_moodtheme 有18486首歌+56个mood/theme多标签
（happy/sad/dark/uplifting...），标签本身带valence语义。用它做辅助多标签
分类任务，和DEAM的VA回归共享backbone联合训练，给共享表征"输血"。

磁盘约束: 音频不按单曲分发，而是100个分片(tar包)，每包~493MB/~185首歌，共
~46GB。**不整包落盘**——用 requests流式GET + tarfile流式读取(fileobj=resp.raw,
mode='r|')，边下载边逐个成员解压；每首歌只临时落盘一个小mp3(~2-8MB)用于
librosa解码，解码完立刻删除。真正持久化到本地的只有计算后的mel特征
(.pt，每个~30KB，比原始音频小~90倍)+标签，18486首全处理完预计~550MB。

预处理是一次性的（跑一次缓存下来，之后训练多少epoch都直接读本地缓存，不用
重新联网）。用 shard 级"完成标记"支持中断后resume，不会因为之前反复被kill
的教训重演而丢进度。

用法:
    python data/mtg_jamendo.py --shards 0-19          # 预处理分片0-19
    python data/mtg_jamendo.py --shards 0-99          # 全量(46GB下载，较慢)
"""

import os
import sys
import csv
import io
import glob
import random
import argparse
import tarfile
import tempfile
import time

import torch
import requests
from torch.utils.data import Dataset, DataLoader

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

from utils.audio import load_audio, mel_spectrogram, aggregate_frames

META_DIR = os.path.join(_root, "data", "mtg_meta")
TSV_PATH = os.path.join(META_DIR, "autotagging_moodtheme.tsv")
TAGS_PATH = os.path.join(META_DIR, "moodtheme.txt")
CACHE_DIR = os.path.join(_root, "data", "mtg_cache")

SHARD_URL = (
    "https://cdn.freesound.org/mtg-jamendo/autotagging_moodtheme/"
    "audio-low/autotagging_moodtheme_audio-low-{shard:02d}.tar"
)

SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
AGGREGATION = 21
MAX_DURATION = 30.0
TARGET_FRAMES = 61  # 30s / (21*512/22050) ≈ 61.5，与config注释里"30s→61帧"一致


def load_tags() -> list:
    tags = []
    with open(TAGS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tags.append(line)
    return tags


def load_metadata() -> dict:
    """返回 {track_id: {"path": str, "tags": [str,...]}}"""
    meta = {}
    with open(TSV_PATH, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            track_id = int(row[0].replace("track_", ""))
            path = row[3]
            tags = row[5:]  # 从第6列开始都是TAGS（可能不止一个，tab分隔）
            meta[track_id] = {"path": path, "tags": tags}
    return meta


def shard_of(track_id: int) -> int:
    return track_id % 100


def _extract_mel(mp3_bytes: bytes) -> torch.Tensor:
    """临时落盘解码，解码完立刻删除临时文件"""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(mp3_bytes)
        tmp_path = tmp.name
    try:
        wav, _ = load_audio(tmp_path, target_sr=SAMPLE_RATE, max_duration=MAX_DURATION)
        mel = mel_spectrogram(wav, sr=SAMPLE_RATE, n_mels=N_MELS,
                               n_fft=N_FFT, hop_length=HOP_LENGTH)
        mel = aggregate_frames(mel, aggregation=AGGREGATION)
        T = mel.shape[-1]
        if T > TARGET_FRAMES:
            mel = mel[..., :TARGET_FRAMES]
        elif T < TARGET_FRAMES:
            mel = torch.nn.functional.pad(mel, (0, TARGET_FRAMES - T))
        return mel
    finally:
        os.remove(tmp_path)


def preprocess_shard(shard_id: int, meta: dict, tags: list, cache_dir: str) -> int:
    """
    流式下载分片tar，逐个成员解压+计算mel+存缓存。返回成功处理的曲目数。
    已有"分片完成"标记则跳过（resume支持）。
    """
    done_marker = os.path.join(cache_dir, f".shard_{shard_id:02d}.done")
    if os.path.exists(done_marker):
        print(f"[跳过] shard {shard_id:02d} 已处理过")
        return 0

    wanted = {tid: info for tid, info in meta.items() if shard_of(tid) == shard_id}
    if not wanted:
        open(done_marker, "w").close()
        return 0

    url = SHARD_URL.format(shard=shard_id)
    print(f"[shard {shard_id:02d}] 需要 {len(wanted)} 首，流式下载: {url}")

    tag_to_idx = {t: i for i, t in enumerate(tags)}
    n_ok, n_fail = 0, 0
    t0 = time.time()

    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with tarfile.open(fileobj=resp.raw, mode="r|") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                # 成员名形如 "48/948.low.mp3"
                base = os.path.basename(member.name)
                if not base.endswith(".low.mp3"):
                    continue
                try:
                    track_id = int(base[: -len(".low.mp3")])
                except ValueError:
                    continue

                if track_id not in wanted:
                    continue  # tarfile流式读取会跳过它的数据段，不写盘

                f = tar.extractfile(member)
                if f is None:
                    n_fail += 1
                    continue
                mp3_bytes = f.read()

                try:
                    mel = _extract_mel(mp3_bytes)
                except Exception as e:
                    print(f"  [WARN] track {track_id} 解码失败: {e}")
                    n_fail += 1
                    continue

                label = torch.zeros(len(tags))
                for t in wanted[track_id]["tags"]:
                    if t in tag_to_idx:
                        label[tag_to_idx[t]] = 1.0

                torch.save(
                    {"mel": mel, "label": label, "track_id": track_id},
                    os.path.join(cache_dir, f"{track_id}.pt"),
                )
                n_ok += 1

    open(done_marker, "w").close()
    elapsed = time.time() - t0
    print(f"[shard {shard_id:02d}] 完成: 成功{n_ok} 失败{n_fail} 耗时{elapsed:.0f}s")
    return n_ok


def preprocess_shards(shard_ids: list, cache_dir: str = CACHE_DIR):
    os.makedirs(cache_dir, exist_ok=True)
    meta = load_metadata()
    tags = load_tags()
    print(f"[MTG] 元数据: {len(meta)} 首, {len(tags)} 个标签")

    total_ok = 0
    for sid in shard_ids:
        total_ok += preprocess_shard(sid, meta, tags, cache_dir)
    print(f"[MTG] 本次共处理 {total_ok} 首新曲目 (累计缓存见 {cache_dir})")


class MTGMoodDataset(Dataset):
    """从预处理缓存(.pt文件)读取mel特征+多标签，不依赖网络/原始音频。"""

    def __init__(self, cache_dir: str, track_ids: list):
        self.cache_dir = cache_dir
        self.track_ids = track_ids

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, idx):
        track_id = self.track_ids[idx]
        data = torch.load(
            os.path.join(self.cache_dir, f"{track_id}.pt"),
            weights_only=True,
        )
        return {"mel": data["mel"], "label": data["label"], "track_id": track_id}


def compute_pos_weight(cache_dir: str, track_ids: list, num_tags: int,
                        max_weight: float = 30.0) -> torch.Tensor:
    """
    按训练集算每个标签的 (负样本数/正样本数)，用于BCE的pos_weight。
    支持缓存 pos_weight.pt 加速启动。
    """
    cache_path = os.path.join(cache_dir, ".pos_weight.pt")
    if os.path.exists(cache_path):
        cached = torch.load(cache_path, weights_only=True)
        if cached.get("track_ids") == track_ids:
            return cached["weight"]

    counts = torch.zeros(num_tags)
    for tid in track_ids:
        data = torch.load(os.path.join(cache_dir, f"{tid}.pt"), weights_only=True)
        counts += data["label"]
    pos = counts.clamp(min=1)
    neg = len(track_ids) - counts
    weight = (neg / pos).clamp(max=max_weight)

    torch.save({"track_ids": track_ids, "weight": weight}, cache_path)
    return weight


def create_mood_dataloaders(
    cache_dir: str = CACHE_DIR,
    batch_size: int = 16,
    num_workers: int = 0,
    val_ratio: float = 0.15,
    seed: int = 42,
):
    """
    Returns: train_loader, val_loader, train_dataset, val_dataset, tags

    只用当前已缓存的曲目（增量式——预处理跑得越多，下次调用可用的数据越多，
    不需要等全部100个分片处理完才能开始联合训练）。
    """
    tags = load_tags()
    cached = sorted(
        int(f[:-3]) for f in os.listdir(cache_dir)
        if f.endswith(".pt") and f[:-3].isdigit()
    )
    rng = random.Random(seed)
    shuffled = cached[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    val_ids = sorted(shuffled[:n_val])
    train_ids = sorted(shuffled[n_val:])

    train_dataset = MTGMoodDataset(cache_dir, train_ids)
    val_dataset = MTGMoodDataset(cache_dir, val_ids)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
    )

    pos_weight = compute_pos_weight(cache_dir, train_ids, len(tags))

    print(f"[MTG] 缓存可用 {len(cached)} 首 -> train {len(train_ids)}, val {len(val_ids)}")
    print(f"[MTG] pos_weight (clip=30): min={pos_weight.min():.1f} "
          f"median={pos_weight.median():.1f} max={pos_weight.max():.1f}")

    return train_loader, val_loader, train_dataset, val_dataset, tags, pos_weight


def parse_shard_range(s: str) -> list:
    if "-" in s:
        lo, hi = s.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=str, default="0-19",
                        help="分片范围，如 '0-19' 或单个 '5'")
    parser.add_argument("--cache_dir", type=str, default=CACHE_DIR)
    args = parser.parse_args()

    shard_ids = parse_shard_range(args.shards)
    preprocess_shards(shard_ids, args.cache_dir)
