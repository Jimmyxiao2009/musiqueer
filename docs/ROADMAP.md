# MusicUnderstand 3.0 — 完整执行计划

> 维护者: Aldrich Voss
> 更新: 2026-05-23

---

## 一、数据下载清单

### 📥 需要下载的数据集

| # | 数据集 | 大小 | 用途 | 来源 | 优先级 |
|---|--------|------|------|------|--------|
| 1 | **FMA Small** | 7.5 GB | 体裁分类（16类，8000首） | Kaggle (`aaronyim/fma-small`) | 🔴 高 |
| 2 | **MusicCaps 音频** | ~10 GB | 真实 Audio-Text 配对训练 | YouTube (via yt-dlp) | 🔴 高 |
| 3 | **MedleyDB** | ~15 GB | 乐器检测 / 多轨分离 | Zenodo/Figshare | 🟡 中 |
| 4 | **GTZAN** | 1.2 GB | 体裁分类（10类） | H800 `/root/autodl-tmp/music_data/gtzan/` | 🟢 已有 |
| 5 | **Music4All-Onion** | 少量 | 元数据 + YouTube ID 列表 | GitHub | 🟢 已有 |
| 6 | **MTG-Jamendo** | 少量 | 体裁+情绪+乐器标签 | GitHub | 🟢 已有 |
| 7 | **DEAM** | 1.3 GB | 情绪标注 VA | H800 `/root/data/deam/` | 🟢 已有 |
| 8 | **Memo2496** | 少量 | VA 情绪标签 | Figshare | 🟢 已有 |
| 9 | **Bilibili 简要分析** | ~300 MB | 94首音频 + 曲风标签 | 本地 `data/bili_analysis/` | 🟢 已有 |

---

### 📝 当前本地已下载

```
H:\musicunderstand2\data\
├── merged/               ✅ 10476条合并训练集
├── music_caps_enhanced   ✅ 8521条描述
├── memo2496/             ✅ VA标签
├── bili_analysis/
│   ├── audio/             ✅ 94首MP3
│   ├── temp/              ✅ 14个视频
│   └── video_list.json    ✅ 18条描述
├── jamendo_*.tsv          ✅ 元数据
├── music4all_tracks.csv   ✅ 元数据
├── deam/DEAM_audio.zip    ✅ 未解压
└── fma_small/             ⏳ 下载中(后台)
```

---

## 二、预处理管线

### 2.1 GTZAN 体裁数据 (已就绪，在 H800)
- 路径: `/root/autodl-tmp/music_data/gtzan/genres/`
- 1000 个 wav，10 类
- 需要: 计算 Mel 频谱缓存
- 脚本: `data/gtzan_dataset.py` ✅

### 2.2 DEAM 情绪数据
- 本地 `DEAM_audio.zip` 未解压
- 操作: 解压 → 计算 Mel 缓存 → 用 DEAM 音频特征
- 脚本: `data/deam_dataset.py` ✅

### 2.3 FMA 体裁数据 (下载中)
- 8000 首 mp3，16 类
- 操作: 解压 → 配元数据 → 计算 Mel 缓存 → 加入训练管线
- 需创建: `data/fma_dataset.py`

### 2.4 Bilibili 视频 OCR
- 14 个 MP4 视频，含曲风字幕
- 操作: 每隔 1s 抽帧 → OCR 提取曲风/调性信息 → 生成标签
- 需安装: `pip install easyocr pillow`
- 需创建: `data/bili_ocr.py`

### 2.5 MusicCaps 音频下载
- 5521 条描述，对应 YouTube 视频
- 操作: 用 yt-dlp 批量下载 → 截取片段 → 计算 Mel 缓存
- 注意: YouTube 需代理或 cookies
- 需创建: `data/download_musiccaps_audio.py`

### 2.6 Memo2496 (已就绪)
- 标签已下载 `data/memo2496/`
- 需创建: `data/memo_dataset.py`

---

## 三、训练计划

### Phase 1: 基础能力 (已完成 ✅)
| 步骤 | 模型 | 数据 | 状态 |
|------|------|------|------|
| 1a | Backbone + Emotion Head | DEAM | ✅ 已训 |
| 1b | Freeze Backbone 微调 | DEAM | ✅ 已训 |
| 1c | Language Decoder | MusicCaps | ✅ PPL 2.3 |
| 1d | 本地增强训练 | 10476 条 | ✅ 9 epoch |

### Phase 2: 体裁 + 乐器 (数据到位后)
| 步骤 | 模型 | 数据 | 依赖 |
|------|------|------|------|
| 2a | Genre Head | GTZAN + FMA | FMA 下完 |
| 2b | Instrument Head | MedleyDB | MedleyDB 下完 |
| 2c | 多任务联合训练 | 全部 | 2a + 2b |

### Phase 3: 语言增强
| 步骤 | 方案 | 数据 | 依赖 |
|------|------|------|------|
| 3a | 真实 Audio-Text 训练 | MusicCaps 音频 | 音频下完 |
| 3b | Phi-3 蒸馏 | Phi-3 GGUF + 10476条 | Phi-3 已就绪 |
| 3c | 生成式合成数据 | Phi-3 → 生成描述 | 3b |
| 3d | 重复问题修复 | 去重 + 多样性增强 | 全数据 |

### Phase 4: 推理部署
| 步骤 | 内容 |
|------|------|
| 4a | 统一推理脚本 `inference_full.py` ✅ |
| 4b | 修复 caption 生成（起始 token）✅ |
| 4c | 流式输出 / Web 界面 |
| 4d | 中文支持（Qwen 蒸馏） |

---

## 四、执行命令速查

### FMA 下载
```bash
# 已启动（后台），查看进度：
python -c "import os; f=os.path.expanduser('~/.cache/kagglehub/datasets/aaronyim/fma-small/1.archive'); print(f'{os.path.getsize(f)/1024/1024:.0f}MB/7570MB' if os.path.exists(f) else 'not found')"
```

### GTX 1650 本地训练
```bash
cd H:\musicunderstand2
python train\train_local.py
```

### 续租 H800 后
```bash
# 1. 装依赖
pip install huggingface_hub

# 2. 下 Qwen3.5 GGUF
# 需要先设置 HF_TOKEN（去 huggingface.co 申请）
huggingface-cli login
huggingface-cli download Qwen/Qwen3.5-2B-GGUF qwen3.5-2b-q4_k_m.gguf

# 3. 跑蒸馏
cd /root/musicunderstand2
python3 -u train/distill.py
```

### B站 OCR
```bash
pip install easyocr pillow
python data/bili_ocr.py
```

---

## 五、依赖清单

```bash
# 基础
pip install torch torchaudio transformers tokenizers librosa pyyaml tqdm matplotlib

# 数据处理
pip install pandas pyarrow kagglehub

# OCR
pip install easyocr pillow

# 蒸馏
pip install llama-cpp-python
# 或 GGUF 推理: pip install ctransformers
```

---

## 六、里程碑

```
□ Phase 1 完成  ✅
□ FMA 下载完毕  ⏳ (后台进行)
□ Phase 2 完成  (需要 FMA + MedleyDB)
□ Phase 3 完成  (蒸馏 + 音频)
□ Phase 4 完成  (推理部署)
```

> 凡事预则立，不预则废。
> — 《礼记·中庸》
> "Preparation leads to success, unpreparedness leads to failure."
