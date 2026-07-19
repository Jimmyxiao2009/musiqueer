"""
音频加载与 Mel 频谱提取工具
适配 MusicUnderstand 2.0 的 21帧聚合策略
"""

import torch
import torchaudio
import librosa
import numpy as np
from typing import Tuple, Optional


def load_audio(
    path: str,
    target_sr: int = 22050,
    max_duration: float = 30.0,
) -> Tuple[torch.Tensor, int]:
    """
    加载音频文件，重采样到目标采样率，截取/填充到 max_duration 秒

    Args:
        path: 音频文件路径 (支持 .wav, .mp3, .m4a, .flac 等)
        target_sr: 目标采样率
        max_duration: 最大时长（秒），超出截取，不足填充0

    Returns:
        wav: [1, T] 单声道音频张量
        sr: 实际采样率
    """
    # 用 torchaudio 加载（支持格式更多）
    try:
        wav, sr = torchaudio.load(path)
    except Exception:
        # fallback 到 librosa
        y, sr = librosa.load(path, sr=None, mono=True)
        wav = torch.from_numpy(y).unsqueeze(0).float()
        sr = sr

    # 混音到单声道
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # 重采样
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        wav = resampler(wav)
        sr = target_sr

    # 截取或填充
    max_samples = int(target_sr * max_duration)
    if wav.shape[-1] > max_samples:
        wav = wav[..., :max_samples]
    elif wav.shape[-1] < max_samples:
        pad = max_samples - wav.shape[-1]
        wav = torch.nn.functional.pad(wav, (0, pad))

    return wav, sr


def mel_spectrogram(
    wav: torch.Tensor,
    sr: int = 22050,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    f_min: int = 0,
    f_max: Optional[int] = None,
) -> torch.Tensor:
    """
    从音频波形提取 Mel 频谱图

    Args:
        wav: [1, T] 音频张量
        sr: 采样率
        n_mels: Mel频带数
        n_fft: FFT窗口大小
        hop_length: 帧移（样本数）
        f_min: 最低频率
        f_max: 最高频率（默认 sr/2）

    Returns:
        mel: [1, n_mels, T_mel] Mel频谱
    """
    if f_max is None:
        f_max = sr // 2

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        power=2.0,
    )

    mel = mel_transform(wav)  # [1, n_mels, T_mel]

    # 转 dB 尺度（更符合人耳感知）
    mel = torchaudio.transforms.AmplitudeToDB(stype="power")(mel)

    # 归一化到 [0, 1]
    mel_min = mel.min()
    mel_max = mel.max()
    if mel_max > mel_min:
        mel = (mel - mel_min) / (mel_max - mel_min)

    return mel


def aggregate_frames(
    mel: torch.Tensor,
    aggregation: int = 21,
) -> torch.Tensor:
    """
    沿时间维度聚合 Mel 帧，降低时间分辨率
    使用平均池化将 aggregation 帧合并为一个时间步

    Args:
        mel: [1, n_mels, T_mel] Mel频谱
        aggregation: 聚合帧数（默认21帧 ≈ 0.5秒）

    Returns:
        mel_agg: [1, n_mels, T_agg] 聚合后的Mel频谱
    """
    # 如果时间维度不是 aggregation 的整数倍，先padding
    T = mel.shape[-1]
    remainder = T % aggregation
    if remainder != 0:
        pad_len = aggregation - remainder
        mel = torch.nn.functional.pad(mel, (0, pad_len))

    # AvgPool1d 做时间维聚合
    pooled = torch.nn.functional.avg_pool1d(
        mel.squeeze(0),  # [n_mels, T]
        kernel_size=aggregation,
        stride=aggregation,
    ).unsqueeze(0)  # [1, n_mels, T_agg]

    return pooled


def extract_features(
    path: str,
    sr: int = 22050,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    aggregation: int = 21,
    max_duration: float = 30.0,
    target_frames: int = 60,
) -> torch.Tensor:
    """
    完整特征提取管线：加载 → Mel → 聚合 → 对齐

    Args:
        path: 音频路径
        sr, n_mels, n_fft, hop_length: 频谱参数
        aggregation: 帧聚合数
        max_duration: 最大时长(秒)
        target_frames: 输出目标帧数

    Returns:
        features: [1, n_mels, target_frames] 对齐后的Mel频谱
    """
    wav, _ = load_audio(path, sr, max_duration)
    mel = mel_spectrogram(wav, sr, n_mels, n_fft, hop_length)
    mel = aggregate_frames(mel, aggregation)

    # 对齐到 target_frames
    T = mel.shape[-1]
    if T > target_frames:
        mel = mel[..., :target_frames]
    elif T < target_frames:
        mel = torch.nn.functional.pad(mel, (0, target_frames - T))

    return mel
