"""
DEAM Dataset loader — 补写缺失模块 (2026-07-11 夜, 原始版本在云端丢失后复原)

依据 docs/PROGRESS.md + config/default.yaml 记录的规格复原:
  - 标注: annotations/annotations averaged per song/dynamic (per second annotations)/
    {valence,arousal}.csv，列 song_id, sample_15000ms, sample_15500ms, ...
    (0.5s 步长，标注从第15秒开始；大多数歌曲45秒，故约60个有效值列)
  - 音频: audio/{song_id}.mp3（扁平目录）
  - target_frames=30 → 每段 15 秒 (30 帧 × 0.5秒)，与 PROGRESS.md "每段时长15秒"一致
  - 15秒滑窗、50%重叠(stride=7.5s)，对 ~45秒歌曲产生 ~5 段，
    与记录的 "1,513首训练歌曲 -> 7,565段"（约5段/歌）吻合
  - 按歌曲 85/15 切分 train/val (同曲片段不跨集，避免泄漏), seed=42
  - delta=中心差分, var=5帧滑动窗口方差 (对齐 train_emotion.py 的 loss 定义)
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import librosa

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from data.dsp_pseudo_labels import compute_tension_complexity, load_stats as _load_dsp_stats

SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FRAME_AGG = 21
TARGET_FRAMES = 30
STEP_SEC = 0.5
SEGMENT_SEC = TARGET_FRAMES * STEP_SEC       # 15.0
STRIDE_SEC = SEGMENT_SEC / 2                 # 7.5 (50% overlap)


def _load_dynamic_csv(path):
    # low_memory=False 会让pandas的C解析器在此CSV的少数不规整行上抛
    # "Buffer overflow" 错误；默认分块解析能容忍这些行 (仅产生dtype警告)，
    # 随后的逐列 to_numeric 强制转换负责清理结果。
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["song_id"] = df["song_id"].astype(int)
    df = df.set_index("song_id")
    # 部分列含非数值占位符 (如字面字符 '-')，而非真正的空值，
    # pd.isna() 检测不到 -> 强制转数值, 无法解析的一律变NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _song_annotated_duration(row):
    """最后一个非NaN列对应的时刻(秒)"""
    valid = row.dropna()
    if len(valid) == 0:
        return 0.0
    last_ms = int(valid.index[-1].replace("sample_", "").replace("ms", ""))
    return last_ms / 1000.0


def _rolling_var(x, win=5):
    half = win // 2
    out = np.zeros_like(x)
    for i in range(len(x)):
        lo, hi = max(0, i - half), min(len(x), i + half + 1)
        out[i] = x[lo:hi].var()
    return out


def _centered_delta(x):
    """中心差分，边界用前向/后向补齐 (跟V/A的delta计算方式一致)"""
    delta = np.zeros_like(x)
    if len(x) > 1:
        delta[1:-1] = (x[2:] - x[:-2]) / 2.0
        delta[0], delta[-1] = x[1] - x[0], x[-1] - x[-2]
    return delta


class DEAMDataset(Dataset):
    def __init__(self, root_dir, valence_df, arousal_df, segments):
        self.audio_dir = os.path.join(root_dir, "audio")
        self.valence_df = valence_df
        self.arousal_df = arousal_df
        self.segments = segments  # list of (song_id, start_sec)
        self.dsp_stats = _load_dsp_stats()
        # T/C伪标签对每个(song_id, start_sec)是确定性DSP函数，缓存到磁盘：
        # 现算会把epoch从~13min拖慢到~21min，缓存后2轮起恢复原速。
        # 全量~8000段×[2,30]float32，缓存总量约2MB，可随时整目录删掉重建
        self.dsp_cache_dir = os.path.join(root_dir, "dsp_cache")
        os.makedirs(self.dsp_cache_dir, exist_ok=True)

    def _tension_complexity(self, song_id, start_sec, y):
        cache_path = os.path.join(self.dsp_cache_dir,
                                  f"{song_id}_{start_sec:.1f}.npy")
        if os.path.exists(cache_path):
            try:
                arr = np.load(cache_path)
                return arr[0], arr[1]
            except Exception:
                pass  # 缓存损坏(如上次写一半被kill)则重算覆盖
        tension, complexity = compute_tension_complexity(
            y, SAMPLE_RATE, FRAME_AGG, TARGET_FRAMES, stats=self.dsp_stats)
        tmp = cache_path + ".tmp.npy"  # 以.npy结尾，np.save才不会再追加后缀
        np.save(tmp, np.stack([tension, complexity]))
        os.replace(tmp, cache_path)   # 原子替换，防止写一半被kill留下坏缓存
        return tension, complexity

    def __len__(self):
        return len(self.segments)

    def _extract_targets(self, song_id, start_sec):
        v_row = self.valence_df.loc[song_id]
        a_row = self.arousal_df.loc[song_id]

        v_vals = np.full(TARGET_FRAMES, np.nan, dtype=np.float32)
        a_vals = np.full(TARGET_FRAMES, np.nan, dtype=np.float32)
        for i in range(TARGET_FRAMES):
            col_ms = int(round((start_sec + i * STEP_SEC) * 1000 / 500) * 500)
            col = f"sample_{col_ms}ms"
            if col in v_row.index and not pd.isna(v_row[col]):
                v_vals[i] = v_row[col]
            if col in a_row.index and not pd.isna(a_row[col]):
                a_vals[i] = a_row[col]

        mask = (~np.isnan(v_vals)) & (~np.isnan(a_vals))
        v_vals = np.nan_to_num(v_vals, nan=0.0)
        a_vals = np.nan_to_num(a_vals, nan=0.0)

        v_delta = np.zeros_like(v_vals)
        a_delta = np.zeros_like(a_vals)
        if TARGET_FRAMES > 1:
            v_delta[1:-1] = (v_vals[2:] - v_vals[:-2]) / 2.0
            a_delta[1:-1] = (a_vals[2:] - a_vals[:-2]) / 2.0
            v_delta[0], v_delta[-1] = v_vals[1] - v_vals[0], v_vals[-1] - v_vals[-2]
            a_delta[0], a_delta[-1] = a_vals[1] - a_vals[0], a_vals[-1] - a_vals[-2]

        v_var = _rolling_var(v_vals)
        a_var = _rolling_var(a_vals)

        emotion = np.zeros((TARGET_FRAMES, 12), dtype=np.float32)
        emotion[:, 0], emotion[:, 1], emotion[:, 2] = v_vals, v_delta, v_var
        emotion[:, 3], emotion[:, 4], emotion[:, 5] = a_vals, a_delta, a_var
        # dims 6-11 (Tension/Complexity): 先占位0，__getitem__ 里用DSP伪标签(见
        # dsp_pseudo_labels.py)覆盖填入，跟V/A共用同一份mask(基于DEAM标注覆盖率)

        return emotion, mask.astype(np.float32)

    def __getitem__(self, idx):
        song_id, start_sec = self.segments[idx]
        mp3_path = os.path.join(self.audio_dir, f"{song_id}.mp3")

        try:
            y, _ = librosa.load(mp3_path, sr=SAMPLE_RATE, mono=True,
                                offset=start_sec, duration=SEGMENT_SEC)
        except Exception as e:
            print(f"[WARN] load failed song={song_id} t={start_sec}: {e}")
            y = np.zeros(int(SAMPLE_RATE * SEGMENT_SEC), dtype=np.float32)

        target_samples = int(SAMPLE_RATE * SEGMENT_SEC)
        if len(y) < target_samples:
            y = np.pad(y, (0, target_samples - len(y)))
        elif len(y) > target_samples:
            y = y[:target_samples]

        mel = librosa.feature.melspectrogram(
            y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
            n_mels=N_MELS, power=2.0)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mn, mx = mel_db.min(), mel_db.max()
        mel_db = (mel_db - mn) / (mx - mn) if mx > mn else np.zeros_like(mel_db)

        mel_t = torch.from_numpy(mel_db).float()  # [n_mels, T]
        T = mel_t.shape[-1]
        rem = T % FRAME_AGG
        if rem != 0:
            mel_t = torch.nn.functional.pad(mel_t, (0, FRAME_AGG - rem))
        mel_agg = torch.nn.functional.avg_pool1d(
            mel_t.unsqueeze(0), kernel_size=FRAME_AGG, stride=FRAME_AGG
        ).squeeze(0)  # [n_mels, T_agg]

        Tagg = mel_agg.shape[-1]
        if Tagg > TARGET_FRAMES:
            mel_agg = mel_agg[:, :TARGET_FRAMES]
        elif Tagg < TARGET_FRAMES:
            mel_agg = torch.nn.functional.pad(mel_agg, (0, TARGET_FRAMES - Tagg))

        emotion, mask = self._extract_targets(song_id, start_sec)

        tension, complexity = self._tension_complexity(song_id, start_sec, y)
        emotion[:, 6] = tension
        emotion[:, 7] = _centered_delta(tension)
        emotion[:, 8] = _rolling_var(tension)
        emotion[:, 9] = complexity
        emotion[:, 10] = _centered_delta(complexity)
        emotion[:, 11] = _rolling_var(complexity)

        return {
            "mel": mel_agg.unsqueeze(0),           # [1, n_mels, TARGET_FRAMES]
            "emotion": torch.from_numpy(emotion),  # [TARGET_FRAMES, 12]
            "mask": torch.from_numpy(mask),        # [TARGET_FRAMES]
            "song_id": song_id,
        }


def _build_segments(song_ids, duration_map):
    segments = []
    for sid in song_ids:
        dur = duration_map.get(sid, 0.0)
        if dur < SEGMENT_SEC:
            segments.append((sid, 0.0))
            continue
        start = 0.0
        while start + SEGMENT_SEC <= dur + 1e-6:
            segments.append((sid, start))
            start += STRIDE_SEC
    return segments


def create_dataloaders(root_dir, batch_size=16, num_workers=0, val_ratio=0.15,
                       seed=42):
    ann_dir = os.path.join(
        root_dir, "annotations", "annotations averaged per song",
        "dynamic (per second annotations)")
    valence_df = _load_dynamic_csv(os.path.join(ann_dir, "valence.csv"))
    arousal_df = _load_dynamic_csv(os.path.join(ann_dir, "arousal.csv"))

    audio_dir = os.path.join(root_dir, "audio")
    audio_ids = set()
    for p in glob.glob(os.path.join(audio_dir, "*.mp3")):
        try:
            audio_ids.add(int(os.path.splitext(os.path.basename(p))[0]))
        except ValueError:
            pass

    common_ids = sorted(set(valence_df.index) & set(arousal_df.index) & audio_ids)
    print(f"[DEAM] valence={len(valence_df)} arousal={len(arousal_df)} "
          f"audio={len(audio_ids)} -> 交集 {len(common_ids)} 首", flush=True)

    duration_map = {}
    for sid in common_ids:
        dv = _song_annotated_duration(valence_df.loc[sid])
        da = _song_annotated_duration(arousal_df.loc[sid])
        duration_map[sid] = min(dv, da)

    rng = np.random.RandomState(seed)
    ids = np.array(common_ids)
    rng.shuffle(ids)
    n_val = int(len(ids) * val_ratio)
    val_id_set = set(ids[:n_val].tolist())
    train_ids = [i for i in common_ids if i not in val_id_set]
    val_ids = [i for i in common_ids if i in val_id_set]

    train_segments = _build_segments(train_ids, duration_map)
    val_segments = _build_segments(val_ids, duration_map)

    print(f"[DEAM] train songs={len(train_ids)} segments={len(train_segments)}",
          flush=True)
    print(f"[DEAM] val   songs={len(val_ids)} segments={len(val_segments)}",
          flush=True)

    train_dataset = DEAMDataset(root_dir, valence_df, arousal_df, train_segments)
    val_dataset = DEAMDataset(root_dir, valence_df, arousal_df, val_segments)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)

    return train_loader, val_loader, train_dataset, val_dataset


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "D:/Musiclearner/data/deam"
    tl, vl, td, vd = create_dataloaders(root, batch_size=4, num_workers=0)
    batch = next(iter(tl))
    print("mel:", batch["mel"].shape)
    print("emotion:", batch["emotion"].shape)
    print("mask:", batch["mask"].shape, "valid frac:", batch["mask"].mean().item())
    print("song_ids:", batch["song_id"])
