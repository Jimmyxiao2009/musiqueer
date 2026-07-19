"""
Tension/Complexity 伪标签 —— DSP启发式代理指标 (不是人工标注真值)

背景: 主流MER数据集(DEAM/PMEmo/MuSe等)基本只标了Valence/Arousal，调研过带
Tension标注的学术数据集(Eerola & Vuoskoski 2011电影配乐库、EMMA/GEMS-45)，
但规模远小于DEAM(百来首vs.1802首)且音频是有版权顾虑的商业录音，不打算用；
Complexity在音乐情感文献里压根没有对应的标注数据集(它是结构/信息论概念，不是
情感量表会覆盖的维度)。详见 docs/PROGRESS.md "VATC四维训练"一节。

所以这里的路线是: 用librosa能直接算出的DSP特征合成"伪真值"去训练这两个头。
本质是让模型一次forward pass学会逼近一个已知的DSP公式，不是让它学到新的感知
理解 —— 训练"完成"的价值是工程整合(自动踩谱工具一次推理拿到四维信息)，不是
科学意义上的"模型学会了理解紧张感/复杂度"。

Tension_raw    = 0.20*dissonance + 0.45*z(rms) + 0.35*z(flux)
  - dissonance: 1 - chroma与大小调模板(Krumhansl-Kessler profile)的最大相关性
    (调性含糊/不协和 → 与任何清晰大小调模板相关性都低 → dissonance高)
  - rms:  短时能量，强奏/响度突增常伴随紧张感
  - flux: 频谱变化率(onset_strength)，代理"躁动感"
  注意: 原始权重0.45/0.30/0.25导致dissonance主导，战斗/史诗音乐虽然气势恢宏
  但不协和度低，结果Tension偏低。新版降低dissonance权重、提高rms+flux权重，
  让响度和频闪变化(躁动感)更充分地反映"紧张/激烈"的主观感知。

Complexity_raw = 0.40*spec_entropy + 0.30*chroma_entropy + 0.30*rhythmic_irregularity
  - spec_entropy:     功率谱香农熵，频率成分越分散越复杂
  - chroma_entropy:   12音级色度香农熵，同时响的音高越多越复杂
  - rhythmic_irregularity: 该片段内onset间隔的变异系数(CV)，节奏越不规则越复杂
    (整段一个值，广播到片段内所有帧，不是逐帧量——检测节奏规律性本来就需要
    比0.5秒长得多的时间窗)

各子特征先用全局统计z-score再加权组合，组合结果再z-score一次，让最终量级
接近DEAM V/A标注本身的"z-score归一化"范围(约[-1,1])，可以复用同一套loss权重
设计。全局统计来自 calibrate_dsp_pseudo_labels.py 预跑一遍存的
data/dsp_pseudo_label_stats.json，训练时直接读取，不重复计算。
"""

import json
import os

import numpy as np
import librosa

SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512

STATS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "dsp_pseudo_label_stats.json")

RAW_FEATURE_NAMES = ["dissonance", "rms", "flux", "spec_entropy",
                     "chroma_entropy", "rhythmic_irregularity"]

# Krumhansl-Kessler 调性profile (标准调性感知/key-finding算法用的经验权重)
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                           2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                           2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _key_templates():
    templates = [np.roll(_MAJOR_PROFILE, k) for k in range(12)] + \
                [np.roll(_MINOR_PROFILE, k) for k in range(12)]
    templates = np.stack(templates)
    return templates / np.linalg.norm(templates, axis=1, keepdims=True)


_KEY_TEMPLATES = _key_templates()  # [24, 12]


def _key_clarity(chroma: np.ndarray) -> np.ndarray:
    """chroma: [12, frames] (非负) -> [frames] 与24个大小调模板的最大相关性"""
    c = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-10)
    corr = _KEY_TEMPLATES @ c  # [24, frames]
    return corr.max(axis=0)


def extract_raw_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """
    对一段音频提取6个原始DSP子特征，帧率是STFT帧粒度(hop_length=512未聚合)。

    Returns: dict[str, np.ndarray]，每个都是 [n_stft_frames]
    """
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)) ** 2
    n_frames = S.shape[1]

    p = S / (S.sum(axis=0, keepdims=True) + 1e-10)
    spec_entropy = -(p * np.log(p + 1e-10)).sum(axis=0) / np.log(S.shape[0])

    chroma = librosa.feature.chroma_stft(S=S, sr=sr)
    chroma_p = chroma / (chroma.sum(axis=0, keepdims=True) + 1e-10)
    chroma_entropy = -(chroma_p * np.log(chroma_p + 1e-10)).sum(axis=0) / np.log(12)
    dissonance = 1.0 - _key_clarity(chroma)

    rms = librosa.feature.rms(S=np.sqrt(S))[0]

    flux = librosa.onset.onset_strength(S=librosa.power_to_db(S), sr=sr)
    if len(flux) != n_frames:
        flux = np.resize(flux, n_frames)

    onset_frames = librosa.onset.onset_detect(onset_envelope=flux, sr=sr,
                                              hop_length=HOP_LENGTH)
    if len(onset_frames) >= 3:
        iois = np.diff(librosa.frames_to_time(onset_frames, sr=sr, hop_length=HOP_LENGTH))
        cv = float(np.std(iois) / (np.mean(iois) + 1e-8))
    else:
        cv = 0.0
    rhythmic_irregularity = np.full(n_frames, cv, dtype=np.float64)

    return {
        "dissonance": dissonance,
        "rms": rms,
        "flux": flux,
        "spec_entropy": spec_entropy,
        "chroma_entropy": chroma_entropy,
        "rhythmic_irregularity": rhythmic_irregularity,
    }


def _zscore(x: np.ndarray, stats: dict, name: str) -> np.ndarray:
    m, s = stats[name]["mean"], stats[name]["std"]
    return (x - m) / (s + 1e-8)


def compute_composites(raw: dict, stats: dict) -> tuple:
    """把raw子特征z-score后加权组合成tension_raw/complexity_raw(未做二次z-score)"""
    tension = (0.20 * _zscore(raw["dissonance"], stats, "dissonance") +
               0.45 * _zscore(raw["rms"], stats, "rms") +
               0.35 * _zscore(raw["flux"], stats, "flux"))
    complexity = (0.40 * _zscore(raw["spec_entropy"], stats, "spec_entropy") +
                 0.30 * _zscore(raw["chroma_entropy"], stats, "chroma_entropy") +
                 0.30 * _zscore(raw["rhythmic_irregularity"], stats, "rhythmic_irregularity"))
    return tension, complexity


def finalize_composites(tension_raw: np.ndarray, complexity_raw: np.ndarray,
                        stats: dict) -> tuple:
    """对组合分数再做一次z-score，量级对齐DEAM V/A标注(约[-1,1])"""
    t = (tension_raw - stats["tension_composite"]["mean"]) / \
        (stats["tension_composite"]["std"] + 1e-8)
    c = (complexity_raw - stats["complexity_composite"]["mean"]) / \
        (stats["complexity_composite"]["std"] + 1e-8)
    return t, c


def pool_to_frames(x: np.ndarray, frame_agg: int, target_frames: int) -> np.ndarray:
    """跟mel聚合(avg_pool1d, kernel=stride=frame_agg)同样的逻辑，把STFT帧率
    序列降到target_frames粒度，跟DEAM V/A标注帧对齐"""
    n = len(x)
    rem = n % frame_agg
    if rem != 0:
        x = np.pad(x, (0, frame_agg - rem), mode="edge")
    x = x.reshape(-1, frame_agg).mean(axis=1)
    if len(x) > target_frames:
        x = x[:target_frames]
    elif len(x) < target_frames:
        x = np.pad(x, (0, target_frames - len(x)), mode="edge")
    return x


_STATS_CACHE = None


def load_stats(path: str = STATS_PATH) -> dict:
    global _STATS_CACHE
    if _STATS_CACHE is not None:
        return _STATS_CACHE
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到DSP伪标签归一化统计文件: {path}\n"
            f"请先运行: python data/calibrate_dsp_pseudo_labels.py")
    with open(path, encoding="utf-8") as f:
        _STATS_CACHE = json.load(f)
    return _STATS_CACHE


def compute_tension_complexity(y: np.ndarray, sr: int, frame_agg: int,
                               target_frames: int, stats: dict = None) -> tuple:
    """
    对一段音频(如DEAM的15秒片段)算出对齐到target_frames粒度的
    tension_mean/complexity_mean数组，供 deam_dataset.py 填入emotion[:,6]/[:,9]。

    Returns: (tension: [target_frames], complexity: [target_frames])
    """
    if stats is None:
        stats = load_stats()
    raw = extract_raw_features(y, sr)
    tension_raw, complexity_raw = compute_composites(raw, stats)
    tension, complexity = finalize_composites(tension_raw, complexity_raw, stats)
    tension = pool_to_frames(tension, frame_agg, target_frames)
    complexity = pool_to_frames(complexity, frame_agg, target_frames)
    return tension.astype(np.float32), complexity.astype(np.float32)
