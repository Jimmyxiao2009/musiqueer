"""
FMA (Free Music Archive) Small/Medium 流式加载器

方案: HTTP Range + SeekableHTTPFile → zipfile 随机读取 → 流式加载单曲
不下载整个 zip，只按需读取需要的 MP3 片段。

用法:
    python data/fma_streaming.py --subset small --workers 4   # 预处理所有歌曲
    python data/fma_streaming.py --subset medium --limit 5000 # 只处理前5000首
"""

import os
import sys
import io
import json
import time
import argparse
import tempfile
import threading
from pathlib import Path
from typing import Optional

import torch
import requests
import zipfile
import numpy as np
import pandas as pd

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

from utils.audio import load_audio, mel_spectrogram, aggregate_frames

# ── 常量 ──────────────────────────────────────────────────────────────
SAMPLE_RATE = 22050
N_MELS = 128
FRAME_AGG = 21
TARGET_FRAMES = 61           # 30s 对齐 MTG-Jamendo

FMA_URLS = {
    "small": "https://os.unil.cloud.switch.ch/fma/fma_small.zip",
    "medium": "https://os.unil.cloud.switch.ch/fma/fma_medium.zip",
    "large": "https://os.unil.cloud.switch.ch/fma/fma_large.zip",
}
FMA_ZIP_SIZES = {
    "small": 7_679_594_875,   # ~7.2 GB
    "medium": 22_642_419_530, # ~21 GB
    "large": 100_283_587_610, # ~100 GB
}
META_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"

CACHE_DIR = os.path.join(_root, "data", "fma_cache")
META_DIR = os.path.join(_root, "data", "fma_meta")


# ── SeekableHTTPFile ──────────────────────────────────────────────────
class SeekableHTTPFile(io.RawIOBase):
    """
    通过 HTTP Range 请求实现可 seek 的文件对象。
    包装一个远程资源，只按需下载真正读取的部分。

    支持: read(), seek(), tell(), readable()
    不支持: write() / writable() (只读)
    """

    def __init__(self, url: str, total_size: int, session: Optional[requests.Session] = None,
                 max_retries: int = 3):
        self.url = url
        self._size = total_size
        self._pos = 0
        self._session = session or requests.Session()
        self._max_retries = max_retries
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        return self._size

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        with self._lock:
            if whence == io.SEEK_SET:
                self._pos = offset
            elif whence == io.SEEK_CUR:
                self._pos += offset
            elif whence == io.SEEK_END:
                self._pos = self._size + offset
            self._pos = max(0, min(self._pos, self._size))
            return self._pos

    def tell(self) -> int:
        with self._lock:
            return self._pos

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def read(self, size: int = -1) -> bytes:
        if size == -1:
            size = self._size - self._pos
        if size <= 0:
            return b""

        end = min(self._pos + size - 1, self._size - 1)
        headers = {"Range": f"bytes={self._pos}-{end}"}

        for attempt in range(self._max_retries):
            try:
                resp = self._session.get(self.url, headers=headers, timeout=30)
                if resp.status_code in (206, 200):
                    data = resp.content
                    with self._lock:
                        self._pos += len(data)
                    return data
                resp.raise_for_status()
            except Exception as e:
                if attempt == self._max_retries - 1:
                    raise
                time.sleep(1 * (attempt + 1))

    def readinto(self, b: bytearray) -> int:
        data = self.read(len(b))
        n = len(data)
        b[:n] = data
        return n

    def close(self):
        pass  # session 由外部管理


def _build_fma_zip_index(url: str, total_size: int, session: requests.Session) -> dict:
    """
    从远程 zip 读取 Central Directory，建立 文件名 → (offset, size) 索引。
    zip 的目录位于文件末尾，通过一次 Range 请求读取。
    支持 ZIP64 格式 (FMA zip 使用 ZIP64)。

    Returns:
        {filename: {"header_offset": int, "compress_size": int, "file_size": int, "crc": int}}
    """
    # 读取 zip 的 EOCD + 尾部 (ZIP64 定位器也可能在这里)
    # 使用 Suffix Range (bytes=-SIZE) 避免大文件 offset 计算问题
    eocd_size = 65557
    resp = session.get(url, headers={"Range": f"bytes=-{eocd_size}"}, timeout=60)
    resp.raise_for_status()
    tail = resp.content

    # 查找 EOCD signature
    eocd_offset_in_tail = tail.rfind(b"\x50\x4b\x05\x06")
    if eocd_offset_in_tail < 0:
        raise ValueError("Cannot find EOCD in zip")
    eocd = tail[eocd_offset_in_tail:]

    central_dir_offset = int.from_bytes(eocd[16:20], "little")
    central_dir_size = int.from_bytes(eocd[12:16], "little")
    central_dir_entries = int.from_bytes(eocd[10:12], "little")

    # ZIP64 检测: offset = 0xFFFFFFFF 表示使用 ZIP64
    if central_dir_offset == 0xFFFFFFFF or central_dir_entries == 0xFFFF or central_dir_size == 0xFFFFFFFF:
        # 查找 ZIP64 End of Central Directory Locator (PK\x06\x07)
        # 它位于 EOCD 之前
        locator_start = eocd_offset_in_tail - 20
        if locator_start >= 0:
            locator = tail[locator_start:locator_start + 20]
            if locator[:4] == b"\x50\x4b\x06\x07":
                z64_eocd_offset = int.from_bytes(locator[8:16], "little")
                # 用 suffix range 读 ZIP64 EOCD
                # EOCD 在文件尾附近，z64_eocd_offset 通常是个很大的数
                # 直接用 session Range 请求
                r2 = session.get(url, headers={"Range": f"bytes={z64_eocd_offset}-{z64_eocd_offset+55}"}, timeout=60)
                z64_eocd = r2.content
                if z64_eocd[:4] == b"\x50\x4b\x06\x06":
                    central_dir_entries = int.from_bytes(z64_eocd[24:32], "little")
                    central_dir_size = int.from_bytes(z64_eocd[40:48], "little")
                    central_dir_offset = int.from_bytes(z64_eocd[48:56], "little")
                    print(f"[ZIP64] entries={central_dir_entries} offset={central_dir_offset} size={central_dir_size}")

    if central_dir_size > 20 * 1024 * 1024:  # > 20MB 保护
        raise ValueError(f"Central directory too large: {central_dir_size} bytes")

    # 用 Range 请求读取中央目录
    r3 = session.get(url, headers={"Range": f"bytes={central_dir_offset}-{central_dir_offset+central_dir_size-1}"}, timeout=120)
    cd_bytes = r3.content

    # 解析中央目录条目
    index = {}
    pos = 0
    while pos + 46 <= len(cd_bytes):
        sig = cd_bytes[pos:pos+4]
        if sig != b"\x50\x4b\x01\x02":
            pos += 1
            continue
        filename_len = int.from_bytes(cd_bytes[pos+28:pos+30], "little")
        extra_len = int.from_bytes(cd_bytes[pos+30:pos+32], "little")
        comment_len = int.from_bytes(cd_bytes[pos+32:pos+34], "little")

        compress_method = int.from_bytes(cd_bytes[pos+10:pos+12], "little")
        crc = int.from_bytes(cd_bytes[pos+16:pos+20], "little")
        compress_size = int.from_bytes(cd_bytes[pos+20:pos+24], "little")
        file_size = int.from_bytes(cd_bytes[pos+24:pos+28], "little")
        local_header_offset = int.from_bytes(cd_bytes[pos+42:pos+46], "little")

        filename_bytes = cd_bytes[pos+46:pos+46+filename_len]
        filename = filename_bytes.decode("utf-8", errors="replace")

        entry_size = 46 + filename_len + extra_len + comment_len
        pos += entry_size

        index[filename] = {
            "header_offset": local_header_offset,
            "compress_size": compress_size,
            "file_size": file_size,
            "crc": crc,
            "method": compress_method,
        }

    print(f"[FMA] zip index: {len(index)} entries")
    return index


def _extract_mel_from_mp3_bytes(mp3_bytes: bytes, target_sr: int = SAMPLE_RATE,
                                 max_duration: float = 30.0) -> torch.Tensor:
    """从 MP3 bytes 中提取 mel 频谱 (对齐 utils/audio.py 的处理逻辑)"""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(mp3_bytes)
        tmp_path = tmp.name
    try:
        wav, sr = load_audio(tmp_path, target_sr, max_duration)
        mel = mel_spectrogram(wav, sr, n_mels=N_MELS)
        mel = aggregate_frames(mel, FRAME_AGG)
        # 对齐到 TARGET_FRAMES
        T = mel.shape[-1]
        if T > TARGET_FRAMES:
            mel = mel[..., :TARGET_FRAMES]
        elif T < TARGET_FRAMES:
            mel = torch.nn.functional.pad(mel, (0, TARGET_FRAMES - T))
        return mel
    finally:
        os.remove(tmp_path)


def _ensure_metadata():
    """确保 FMA 元数据 (tracks.csv) 已下载并缓存"""
    if not os.path.exists(META_DIR):
        os.makedirs(META_DIR, exist_ok=True)

    tracks_path = os.path.join(META_DIR, "tracks.csv")
    if not os.path.exists(tracks_path):
        print("[FMA] 下载元数据 tracks.csv...")
        resp = requests.get(META_URL, stream=True, timeout=300)
        resp.raise_for_status()

        # 流式解压，只提取 tracks.csv
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        zf.extract("tracks.csv", META_DIR)
        # 可能有子目录
        extracted = os.path.join(META_DIR, "tracks.csv")
        if not os.path.exists(extracted):
            # 找子目录
            for name in zf.namelist():
                if name.endswith("tracks.csv"):
                    zf.extract(name, META_DIR)
                    # 移动到根目录
                    src = os.path.join(META_DIR, name)
                    if src != extracted:
                        import shutil
                        shutil.move(src, extracted)
                    break
        print(f"[FMA] 元数据缓存至 {tracks_path}")

    return tracks_path


def _load_fma_metadata(subset: str) -> pd.DataFrame:
    """
    加载 FMA tracks.csv，过滤出 subset 对应的曲目，返回
    DataFrame 包含: track_id, genre_top, split (training/test/val)
    """
    tracks_path = _ensure_metadata()
    tracks = pd.read_csv(tracks_path, header=[0, 1], low_memory=False)

    # 解压 MultiIndex
    # 注意: 第一列 (Unnamed: 0) 是实际 track_id
    flat_cols = []
    for a, b in tracks.columns:
        if "Unnamed" in str(a):
            # 第一列是 track_id
            flat_cols.append("track_id")
        elif b and "Unnamed" not in str(b):
            flat_cols.append(f"{a}_{b}")
        else:
            flat_cols.append(a)
    tracks.columns = flat_cols

    # FMA small/medium/large 在 set_subset 列中是字符串
    if subset == "small":
        tracks = tracks[tracks["set_split"].isin(["training", "validation", "test"])]
        tracks = tracks[tracks["set_subset"] == "small"]
    elif subset == "medium":
        tracks = tracks[tracks["set_subset"].isin(["small", "medium"])]
    elif subset == "large":
        tracks = tracks[tracks["set_subset"].isin(["small", "medium", "large"])]

    tracks = tracks[tracks["track_genre_top"].notna()].copy()
    tracks["track_id"] = tracks["track_id"].astype(int)
    tracks["genre_top"] = tracks["track_genre_top"].astype(str)
    tracks["split"] = tracks["set_split"].astype(str)

    # 过滤无效 ID
    tracks = tracks[tracks["track_id"] > 0]

    return tracks[["track_id", "genre_top", "split"]]


def _process_one_track(row, index, url, total_size, cache_dir, fma_zip_paths, genre_to_idx):
    """处理单首 FMA 曲目（供线程池调用）"""
    import bz2, zlib
    session = requests.Session()
    tid = int(row["track_id"])
    cache_path = os.path.join(cache_dir, f"{tid}.pt")

    if os.path.exists(cache_path):
        return "skip", tid

    if tid not in fma_zip_paths:
        return "fail", tid

    zip_path = fma_zip_paths[tid]

    try:
        entry = index[zip_path]
        sf = SeekableHTTPFile(url, total_size, session)
        sf.seek(entry["header_offset"])

        local_header = sf.read(30)
        fn_len = int.from_bytes(local_header[26:28], "little")
        extra_len = int.from_bytes(local_header[28:30], "little")
        sf.seek(entry["header_offset"] + 30 + fn_len + extra_len)
        compressed = sf.read(entry["compress_size"])

        if entry["method"] == 0:
            mp3_data = compressed
        elif entry["method"] == 8:
            dec = zlib.decompressobj(-zlib.MAX_WBITS)
            mp3_data = dec.decompress(compressed) + dec.flush()
        elif entry["method"] == 12:
            mp3_data = bz2.decompress(compressed)
        else:
            return "fail", tid

        mel = _extract_mel_from_mp3_bytes(mp3_data)

        label = torch.zeros(len(genre_to_idx))
        label[genre_to_idx[row["genre_top"]]] = 1.0

        torch.save(
            {"mel": mel, "label": label, "genre": row["genre_top"],
             "track_id": tid, "split": row["split"]},
            cache_path,
        )
        return "ok", tid

    except Exception:
        return "fail", tid


def preprocess_fma(subset: str = "small", num_workers: int = 1, limit: int = 0):
    """
    预处理 FMA 数据集：
    1. 建立远程 zip 索引
    2. 按需流式下载每首歌 → mel 缓存
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    url = FMA_URLS[subset]
    total_size = FMA_ZIP_SIZES[subset]
    tracks = _load_fma_metadata(subset)
    print(f"[FMA] {subset}: {len(tracks)} tracks with genre_top")

    if limit > 0:
        tracks = tracks.head(limit)
        print(f"[FMA] limited to {limit} tracks")

    session = requests.Session()
    print(f"[FMA] building zip index...")
    index = _build_fma_zip_index(url, total_size, session)

    fma_zip_paths = {}
    for name in index:
        if name.endswith(".mp3") and not name.startswith("."):
            base = os.path.basename(name)
            try:
                tid = int(base[:-4])
                fma_zip_paths[tid] = name
            except ValueError:
                pass

    genre_to_idx = {g: i for i, g in enumerate(sorted(tracks["genre_top"].unique()))}
    print(f"[FMA] {len(genre_to_idx)} genres")

    # Count what needs processing
    rows_to_process = []
    for _, row in tracks.iterrows():
        tid = int(row["track_id"])
        if os.path.exists(os.path.join(CACHE_DIR, f"{tid}.pt")):
            continue
        if tid in fma_zip_paths:
            rows_to_process.append(row)

    n_skipped = len(tracks) - len(rows_to_process)
    print(f"[FMA] to process: {len(rows_to_process)}, cached: {n_skipped}")

    n_ok, n_fail = 0, 0
    t0 = time.time()

    if num_workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = [pool.submit(
                _process_one_track, row, index, url, total_size,
                CACHE_DIR, fma_zip_paths, genre_to_idx,
            ) for _, row in tracks.iterrows() if not os.path.exists(
                os.path.join(CACHE_DIR, f"{int(row['track_id'])}.pt")
            ) and int(row["track_id"]) in fma_zip_paths]

            for f in as_completed(futures):
                result, tid = f.result()
                if result == "ok":
                    n_ok += 1
                else:
                    n_fail += 1

                if (n_ok + n_fail) % 50 == 0:
                    elapsed = time.time() - t0
                    print(f"  [{subset}] {n_ok} ok / {n_fail} fail, {elapsed:.0f}s")
    else:
        for row in rows_to_process:
            tid = int(row["track_id"])
            result, tid = _process_one_track(
                row, index, url, total_size, CACHE_DIR,
                fma_zip_paths, genre_to_idx,
            )
            if result == "ok":
                n_ok += 1
            else:
                n_fail += 1

            if n_ok > 0 and n_ok % 50 == 0:
                elapsed = time.time() - t0
                print(f"  [{subset}] {n_ok} ok / {n_fail} fail / {n_skipped} skip, {elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"[FMA {subset}] done: {n_ok} ok, {n_fail} fail, {n_skipped} skip, {elapsed:.0f}s")


class FMADataset(torch.utils.data.Dataset):
    """从预处理缓存读取 FMA mel 特征 + genre 标签"""

    def __init__(self, cache_dir: str = None, split: str = None):
        self.cache_dir = cache_dir or CACHE_DIR
        self.split = split
        self.cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith(".pt")]

        if self.split:
            self.cache_files = [
                f for f in self.cache_files
                if self._get_split(f) == split
            ]

        print(f"[FMADataset] {len(self.cache_files)} 样本 (split={split})")

    def _get_split(self, fname: str) -> str:
        try:
            data = torch.load(os.path.join(self.cache_dir, fname), map_location="cpu")
            return data.get("split", "training")
        except Exception:
            return "unknown"

    def __len__(self):
        return len(self.cache_files)

    def __getitem__(self, idx):
        data = torch.load(os.path.join(self.cache_dir, self.cache_files[idx]),
                          map_location="cpu", weights_only=True)
        return {
            "mel": data["mel"],
            "label": data["label"],  # one-hot genre
            "track_id": data.get("track_id", -1),
        }


def create_dataloaders(batch_size: int = 16, split: str = None, num_workers: int = 0):
    """便捷接口: 创建 FMA DataLoader"""
    dataset = FMADataset(split=split)
    if len(dataset) == 0:
        return None
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=(split != "test"),
        num_workers=num_workers, pin_memory=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FMA 流式预处理")
    parser.add_argument("--subset", choices=["small", "medium", "large"], default="small")
    parser.add_argument("--limit", type=int, default=0, help="限制处理数量 (调试)")
    parser.add_argument("--workers", type=int, default=1, help="并行线程数")
    args = parser.parse_args()

    preprocess_fma(subset=args.subset, num_workers=args.workers, limit=args.limit)
