# MusicUnderstand 3.0 — 音乐理解模型

> 一个能在 GTX 1650 上跑的多任务音乐理解模型。
> 情绪分析 + 体裁分类 + 乐器检测 + 语言描述 + Chain of Thought 推理。
>
> 作者: 奥尔德里克·沃斯 (Aldrich Voss), House Eternal
> 日期: 2026-05-23

---

## 🏛️ 项目概述

MusicUnderstand 3.0 是一个基于 Transformer 的多任务音乐理解模型。

### 能力

| 能力 | 说明 | 状态 |
|------|------|------|
| 🎵 情绪分析 | Valence/Arousal 逐帧输出 | ✅ 已训练 (DEAM) |
| 🎸 体裁分类 | 10 类体裁预测 | ✅ 架构就绪 (需 FMA 训练) |
| 🎻 乐器检测 | 10 类多标签检测 | ✅ 架构就绪 (需数据) |
| 💬 语言描述 | 生成自然语言音乐描述 | ✅ 已训练 (PPL 2.3) |
| 🧠 CoT 推理 | 分步推理+工具调用 | ✅ 架构就绪 |

### 架构

```
Mel [B, 1, 128, T]
  ↓
Transformer Backbone (6层 Encoder, d_model=256)
  ↓ multi-scale features
  ├→ Emotion Head → [B, T, 12]  (Valence/Arousal/Tension/Complexity)
  ├→ Genre Head   → [B, 10]     (blues/classical/country/...)
  ├→ Instrument Head → [B, 10]  (piano/drums/bass/...)
  └→ Language Decoder (4层, d_model=256, Cross-Attention + Audio Gate)
       ↓
     [B, T_text, vocab_size]  (GPT-2 tokenizer, 50257 vocab)
```

总参数: ~22.6M (Backbone 5.5M + Decoder 17M + Heads 0.1M)

---

## 📦 数据集

### 已就绪 (本地 H:\musicunderstand2\data\)

| 数据集 | 规模 | 用途 | 路径 |
|--------|------|------|------|
| **MusicCaps 增强** | 8521 条 | 语言描述训练 | `music_caps_enhanced.json` |
| **Memo2496** | 2496 首 | VA 情绪标注 | `memo2496/` |
| **Bilibili 分析** | 94 首音频 | 曲风格签 | `bili_analysis/audio/` |
| **Bilibili 元数据** | 105 条 | 曲风描述 | `bili_analysis/season_all.json` |
| **FMA Small** | 8000 首 | 体裁分类 | `fma_small/` (156 子目录) |
| **DEAM** | 1802 首 | 情绪标注 | `deam/` (需解压) |
| **合并训练集** | 10476 条 | 统一训练 | `merged/unified_training.json` |
| **CoT 数据** | 3000 条 | 推理训练 | `cot_data/cot_training.json` |

### 待下载

| 数据集 | 规模 | 来源 |
|--------|------|------|
| FMA Medium | 22GB / 25000 首 | Kaggle (imsparsh/fma) |
| MusicCaps 音频 | ~10GB | YouTube (需 yt-dlp) |
| MedleyDB | ~15GB | Zenodo |

---

## 🚀 快速开始

### 环境
```bash
pip install torch torchaudio librosa tokenizers tqdm pyyaml matplotlib
pip install pandas kagglehub  # 数据处理
```

### 推理
```bash
# 基础推理 (情绪+体裁+语言描述)
python inference_local.py

# Chain of Thought 推理 (分步分析)
python inference_cot.py
```

### 训练
```bash
# v3.1 改进训练 (Audio Gate + Scheduled Sampling)
python train/train_v3.1.py

# 生成 CoT 数据
python data/generate_cot_data.py
```

---

## 🧠 技术特性 (v3.1)

### Audio Gate
Cross-Attention 中加入音频能量门控，控制解码器对音频特征的依赖程度。

### Multi-scale Backbone
Backbone 输出第 1 层(局部细节) + 第 5-6 层(全局语义)的拼接特征。

### Scheduled Sampling
训练时从 teacher forcing 逐步过渡到模型自生成，解决"一步错步步错"问题。

### Chain of Thought + ReAct
模型分步调用 expert tool (emotion/genre/instrument/tempo)，观察结果后综合推理。

---

## 📁 项目结构

```
H:\musicunderstand2\
├── model/                          # 网络模块
│   ├── backbone.py                 # Transformer Encoder ×6
│   ├── music_understand.py         # 主模型
│   └── experts/
│       ├── emotion_head.py         # 情绪回归头
│       ├── genre_head.py           # 体裁分类头
│       ├── instrument_head.py      # 乐器检测头
│       ├── language_decoder.py     # 语言解码器 (v3.1: Audio Gate)
│       └── react_decoder.py        # ReAct 推理框架
├── train/                          # 训练脚本
│   ├── train_emotion.py            # Step 1 情绪训练
│   ├── train_language.py           # 语言模型训练
│   ├── train_multitask.py          # 多任务训练
│   ├── train_v3.1.py               # v3.1 改进训练
│   └── distill.py                  # 蒸馏脚本
├── data/                           # 数据集
│   ├── merged/                     # 合并训练集
│   ├── cot_data/                   # CoT 数据
│   ├── fma_small/                  # FMA 体裁数据 (8000首)
│   ├── bili_analysis/              # B站音乐分析
│   ├── memo2496/                   # VA 情绪标签
│   └── generate_cot_data.py        # CoT 数据生成器
├── checkpoints/                    # 模型检查点
├── inference_local.py              # 本地推理
├── inference_cot.py                # CoT 推理
├── ROADMAP.md                      # 开发路线图
└── TECH_IMPROVEMENTS.md            # 技术改进方案
```

---

## 📊 训练结果

| 模型 | PPL | 数据 | 说明 |
|------|-----|------|------|
| language_decoder_final.pth | 2.3 | 8521 条 | MusicCaps 增强训练 |
| local_ep9.pth | ~2.0 | 10476 条 | 本地纯文本训练 |
| v3.1_ep*.pth | — | 10476 条 | Audio Gate + SS (训练中) |

---

## 📋 待办

- [ ] FMA Medium 下载 → 训 Genre Head
- [ ] MedleyDB 下载 → 训 Instrument Head
- [ ] H800 续租 → 蒸馏 Phi-3
- [ ] B站视频 OCR → 曲风格签
- [ ] Web 推理界面
- [ ] 中文描述支持

---

> 锲而不舍，金石可镂。
> — 荀子《劝学》
