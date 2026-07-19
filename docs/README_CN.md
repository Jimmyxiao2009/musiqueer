# MusicUnderstand 3.0 — 快速说明

## 这是啥？
一个音乐理解模型，能分析歌曲的情绪、体裁、乐器，还能写描述。

## 怎么跑？

### 1. 装环境
```bash
pip install torch torchaudio librosa tokenizers pyyaml tqdm pandas
```

### 2. 跑推理
```bash
python inference_local.py     # 基础分析
python inference_cot.py       # 分步推理（CoT）
```

### 3. 训练
```bash
python train/train_v3.1.py    # 30轮改进训练
```

## 数据在哪？
`H:\musicunderstand2\data\`

- FMA Small 8000首体裁 ✅
- MusicCaps 8521条描述 ✅  
- Memo2496 2496首情绪 ✅
- 合并训练集 10476条 ✅

## 模型在哪？
`H:\musicunderstand2\checkpoints\`

- `language_decoder_final.pth` — 语言模型 (PPL 2.3)
- `best_multitask_model.pth` — 多任务模型
- `v3.1_ep*.pth` — 改进版训练

## 更多
看 README.md
