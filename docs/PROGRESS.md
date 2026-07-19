# MusicUnderstand 2.0 — 进度追踪

> 维护者: Aldrich Voss, Steward of House Eternal
> 更新: 2026-05-22

---

## 总览

| Stage | 状态 | 说明 |
|-------|------|------|
| Step 1: Transformer + Emotion Head | ✅ **完成** | DEAM 数据集，12维情绪输出 |
| Step 2: Genre / Instrument Head | ⏳ 待开始 | GTZAN 体裁 + 乐器检测 |
| Step 3: 多数据集联合训练 | ⏳ 待开始 | DEAM + Soundtracks + FMA + GTZAN |
| Step 4: LLM Interpreter | 📅 规划中 | 情绪曲线 → 自然语言描述 |

---

## Step 1 完成情况

### 项目结构

```
musicunderstand2/
├── config/default.yaml              # 所有可调参数
├── model/
│   ├── backbone.py                  # Transformer Encoder × 6
│   ├── experts/emotion_head.py      # 12维情绪回归头
│   └── music_understand.py          # 主模型组装
├── data/deam_dataset.py             # DEAM 加载器（2.0 版数据管线）
├── train/train_emotion.py           # 训练脚本
├── inference.py                     # 推理脚本
├── utils/
│   ├── audio.py                     # Mel 频谱 + 21帧聚合
│   └── metrics.py                   # R², RMSE, Pearson r, CCC
├── checkpoints/                     # 模型保存位置
├── requirements.txt
├── README.md
└── PROGRESS.md                      # ← 本文件
```

### 架构变更（1.0 → 2.0）

| 组件 | 1.0 (CNN+LSTM) | 2.0 (Transformer) |
|------|----------------|-------------------|
| **前端** | Conv2D [64→128→256] + (2,1) pool | Conv2D [64→128] + stride=(8,1) 频带压缩 |
| **时序** | BiLSTM (2层, hidden=256) | Transformer Encoder (6层, 8头, d_model=256) |
| **位置编码** | 无（LSTM 自带顺序性） | 正弦位置编码（可处理任意长度） |
| **归一化** | BatchNorm (CNN) | LayerNorm (Pre-LN Transformer) |
| **激活** | ReLU | GELU |
| **输出** | [B, T, 2] (V/A) | [B, T, 12] (V/A/T/C × mean/delta/var) |
| **参数** | ~3.0M | ~7.2M |

### 关键技术决策

#### Pre-LN Transformer 结构
- 每个 Block: LayerNorm → MHA → residual → LayerNorm → FFN → residual
- **为什么**: Post-LN 在深层 Transformer 中容易出现梯度消失，Pre-LN 更稳定，不需要学习率 warmup

#### 正弦位置编码 vs 可学习编码
- 选择正弦而非可学习的 Embedding
- **为什么**: 正弦编码对训练中未见过的长度泛化更好，DEAM 的 45 秒 -> 30 秒截取不会产生未学习的 token 位置

#### 频带压缩策略
- 第一层 Conv2D: kernel=3, stride=1 (保持 128 mel 频带)
- 第二层 Conv2D: kernel=(8,3), stride=(8,1) (压缩到 16 个频带)
- **为什么**: 保留局部频带特征（第一层）后再压缩维度（第二层），相比直接压到 16，保留了更多局部频谱结构

#### 21 帧聚合
- Mel 频谱 hop_length=512 (≈0.023s/帧 @ 22050Hz)
- 每 21 帧用 AvgPool1d 聚合成一个输出点（≈0.5秒）
- **为什么**: 与 DEAM 标注的 0.5s 步长对齐，同时降低 Transformer 序列长度（30秒 → 60帧），减少显存占用

#### 12 维度设计
| 维 | 含义 | 计算方法 |
|----|------|---------|
| 0 (V_mean) | 情绪效价 | 模型直接预测 |
| 1 (V_delta) | 效价变化趋势 | 中心差分，ground truth 从 mean 计算 |
| 2 (V_var) | 效价波动幅度 | 5帧窗口方差，ground truth 从 mean 计算 |
| 3 (A_mean) | 能量水平 | 模型直接预测 |
| 4 (A_delta) | 能量变化趋势 | 同上 |
| 5 (A_var) | 能量波动幅度 | 同上 |
| 6-8 (T_*) | 张力 | 暂未训练（标注来自 Soundtracks） |
| 9-11 (C_*) | 复杂度 | 暂未训练（Phase 3 用伪标签） |

**Loss 权重**: V/A_mean=1.0, delta=0.5, var=0.3（当前的启发式配置）

#### 梯度裁剪 (1.0)
Transformer 比 LSTM 更容易出现梯度尖峰，clip=1.0

### 硬件适配

| 规格 | 数值 |
|------|------|
| GPU | GTX 1650 (4GB VRAM) |
| Batch size | 16 (4GB 显存限制，T=60 时 ~2.8GB 占用) |
| 可尝试 max | batch=32 (4GB 满，不推荐) |
| 数据类型 | FP32 (1650 无 Tensor Core) |
| 预计训练时间 | ~2-4 小时 (100 epochs, ~14K segments) |

### 数据集细节

| 属性 | 数值 |
|------|------|
| 训练歌曲 | 1,513 首 |
| 验证歌曲 | 267 首 |
| 每段时长 | 15 秒（30 帧 × 0.5 秒） |
| 训练片段数 | 7,565 段 |
| 验证片段数 | 1,335 段 |
| V/A 范围 | ~[-0.83, 0.90] (Z-score全局归一化) |

### 已知问题

#### DEAM 标注是 Z-score 而非原始 1-9
- CSV 中的 V/A 值已经是全局 Z-score 归一化（范围约 [-0.83, 0.90]），而非原始 1-9 分
- 所以 EmotionHead 输出**不使用 Sigmoid**，保持无约束线性输出
- 推理时再做 [0, 1] 映射

#### Valence/Arousal CSV 歌曲不一致
- valence.csv 有 1786 首歌，arousal.csv 有 1787 首，交集 1780 首
- 与 mp3 音频（1802 首）取三交集后得到 **1,780 首可用**（85%/15% 切分获得 1,513/267）

### 与 1.0 的性能对比预期

| 指标 | 1.0 (CNN+LSTM) | 2.0 (Transformer) 预期 |
|------|----------------|----------------------|
| V/A R² | ~0.3-0.4 | ~0.4-0.55 (注意力全局视野优势) |
| V/A Pearson r | ~0.55-0.65 | ~0.65-0.78 |
| 长程结构捕捉 | 弱（LSTM 梯度衰减） | 强（注意力直达） |
| 训练速度 | ~2 it/s | ~1-1.5 it/s (6层注意力计算开销) |
| 推理速度 | 快 | 略慢但实时足够 |

---

## 待办

- [x] 项目结构搭建
- [x] Transformer Backbone (6层, Pre-LN)
- [x] 12维 Emotion Head
- [x] DEAM 数据加载器 (21帧聚合)
- [x] 训练脚本 (AdamW + CosineAnnealing)
- [x] 推理脚本 + 可视化
- [x] 首次训练完成并记录结果 (2026-07-11，见下方"Step 1 首次训练结果")
- [ ] 与 1.0 版本对比测试
- [ ] 步进损失权重调节
- [x] 验证 GTX 1650 实际训练速度 (~150s/epoch，4-worker CPU 加载，比预估快得多)

---

## Step 1 首次训练结果 (2026-07-11)

> `data/deam_dataset.py` 在此次会话中补写（原云端版本已丢失，见下方"数据管线复原"）。
> 配置：batch_size=16, num_workers=4, epochs=25 (patience=15 早停)。

**训练在 epoch 19 早停**（最后一次改善在 epoch 4，之后 15 轮 val_loss 未破纪录）。
最佳 checkpoint：`checkpoints/best_model.pth`（val_loss=0.0935，来自 epoch 4）。
完整曲线：`checkpoints/training_curves.png`。

| Epoch | Val Loss | V_mean R² | V_mean PearsonR | A_mean R² | A_mean PearsonR |
|:-----:|:--------:|:---------:|:---------------:|:---------:|:----------------:|
| 5     | 0.1186   | -0.1047   | 0.1562           | 0.3817    | 0.6782            |
| 10    | 0.1014   | -0.0379   | 0.2931           | **0.5277**| **0.7275**        |
| 15    | 0.1066   | -0.1087   | 0.2841           | 0.5115    | 0.7199            |

**Arousal 学得很好**：R²/PearsonR 在 epoch 10 附近达到峰值（0.53 / 0.73），落在本文档
预期区间（R² 0.4-0.55, Pearson r 0.65-0.78）之内，且比 25 epoch 预算更快达到。

**Valence 没学起来**：R² 全程为负（比预测均值还差），PearsonR 峰值仅 0.29。这与音乐
情绪识别文献中常见的不对称一致——Arousal（响度/节奏/频谱能量）是声学层面可直接
提取的物理量，Valence（情绪褒贬）依赖和声/旋律走向等更深的语义理解，天然更难学。

**⚠️ 已知设计缺陷（下次训练前应修）**：`train_emotion.py` 只在 **val_loss（复合
MSE）改善时**保存 checkpoint，不是按 R²/PearsonR（更能代表下游可用性的指标）来选。
本次训练中 epoch 10 的 Arousal 相关性明显优于 epoch 4，但由于 epoch 4 的 val_loss
更低，只有 epoch 4 的权重被保存——**epoch 10 那个相关性更强的模型权重已经丢失，
无法恢复**，`training_curves.png` 里能看到它曾经存在，但拿不到它的checkpoint。
建议下次训练加一个基于 `A_mean_PearsonR`（或 V+A 综合分）的独立最佳checkpoint
保存逻辑，与 val_loss 版本并存。

### 数据管线复原说明

`data/deam_dataset.py` 在本次会话中从 `docs/PROGRESS.md`/`config/default.yaml` 记录
的规格复原（原始版本在云端 H800 实例上编写，未同步回本地，本地副本已丢失）。复原后
跑出的 train/val 歌曲数（1513/267）与本文档"数据集细节"章节记录的历史数字精确一致，
交叉验证了切分逻辑的正确性；片段数（6805/1162 train/val）与历史记录（7565/1335）
接近但不完全相同（滑窗 stride=7.5s 是本次合理选定的复原参数，非精确复刻已丢失的
原始实现）。复原时发现并修复两个数据坑：
1. DEAM 标注 CSV 中少数格子是字面字符 `'-'`（非真正空值），需 `pd.to_numeric(errors="coerce")`
   逐列强制转换，`pd.isna()` 检测不到这类值。
2. `pd.read_csv(..., low_memory=False)` 在此文件上会崩溃（`Buffer overflow`），
   必须用默认的分块解析。

---

## Step 1 重训会话 (2026-07-11 白天/深夜)

### 目的与背景

上一次会话（本文档"首次训练结果"章节）发现 `train_emotion.py` 只按 val_loss
选 checkpoint、不按相关性选，导致表现最好的 epoch 权重可能丢失。本次会话的
目标：修复这个bug，重训一轮，拿到一个可靠留得住的最佳checkpoint。

### 代码修复

`train/train_emotion.py`：每个epoch额外计算 `(V_mean_PearsonR + A_mean_PearsonR)/2`
作为综合相关性分数，创新高时单独存 `checkpoints/best_corr_model.pth`，与按
val_loss 选的 `checkpoints/best_model.pth` 并存，不再相互覆盖。

新增 `train/evaluate_emotion.py`：独立评估脚本，加载任意checkpoint在本地DEAM
验证集上跑完整指标，不依赖训练循环。用它验证过 CPU/GPU 跑出的指标完全一致
（确定性验证通过），也用它复测了历史 checkpoint 的真实效果（见下）。

### 环境问题（本次会话修复）

1. Python环境的 `torch` 一度是 CPU-only 版 (`2.11.0+cpu`)，与之前训练用的
   CUDA环境不一致（原因不明，可能环境被覆盖）。重装为 `torch==2.6.0+cu124`
   （官方 cu124 索引目前最高只到 2.6.0，装不到 2.11.0 的 CUDA 版）。
2. 环境缺 `matplotlib`/`pandas`，已补装。
3. Windows下 Python stdout 重定向到文件时默认用 GBK 编码，脚本里的"✓"符号
   会导致 `UnicodeEncodeError` 崩溃——运行训练脚本必须设置
   `PYTHONIOENCODING=utf-8`。

### 复测历史checkpoint的真实效果

用新写的 `evaluate_emotion.py` 精确复测了 `best_model_run1_epoch4.pth`
（上次训练val_loss最低的epoch4，已备份保留）：
**Arousal R²=0.575，PearsonR=0.76，CCC=0.72**——比本文档之前记录的"epoch10峰
值"(R²=0.53)还好。说明上次会话"丢失了最好的checkpoint"的担忧其实是误判：
epoch4从未被打印进日志表格（表格只记录了5/10/15），实际上一直安然保存着，
没有丢失。**Valence R²=0.016**，基本没学到，比文档原先描述的"全程为负"略好
但同样接近无效。

### 本次重训进展与反复中断

用 `--resume` 从epoch4续训，目标是走完25epoch预算验证是否有更优checkpoint。
过程中被后台任务机制连续中断了3次（每次都无Python报错堆栈，进程直接消失）：

| 尝试 | num_workers | 运行时长 | 结果 |
|-----|:-----------:|:--------:|------|
| 2 | 4 | ~15-20min | 被kill，epoch4后无新进展 |
| 3 | 4 | ~53min | 被kill，到epoch8 (val_loss=0.0938, corr=0.5322) |
| 5 | 0 (改为单进程排查OOM可能性) | ~32min | 仍被kill，到epoch10 (val_loss=0.1059, corr=0.5191) |

排查过：系统空闲内存一度探底到877MB（`num_workers=4`每epoch重新spawn 4个
worker进程、各自加载torch+CUDA，可能造成内存尖峰），改 `num_workers=0`后
内存确实更稳定，但仍然被杀——说明OOM不是唯一/确定原因。查了Windows系统
日志，发现频繁的"系统时间变更"事件（`Kernel-General` ID 16，每1-2分钟一
对）和一次会话转换事件（`Kernel-Power` ID 566，SessionUnlock），怀疑与
运行环境本身的挂起/恢复机制有关，但没有精确对应到每次kill的时间点，**未能
100%确认根因**。每次都用 `--resume` 从最新checkpoint接上，没有丢失训练
进度本身。

### 决定：停在这里，不继续追更高epoch

复测 `best_corr_model.pth`（本次重训到epoch10的最佳相关性checkpoint）：
Arousal R²=0.4515，PearsonR=0.7359，CCC=0.7135；Valence R²=-0.0218，
PearsonR=0.3024。**epoch10的Arousal R²反而不如epoch4（0.45 vs 0.575）**，
PearsonR/CCC相近。这印证了一个规律：这个模型在DEAM上的学习主要发生在前几个
epoch，之后在相关性指标上是震荡而非单调提升的。继续追更高epoch面对反复
中断的不稳定环境，边际收益已经不高。

**结论**：`checkpoints/best_model_run1_epoch4.pth` 仍是目前样本内最好的
单一checkpoint（Arousal R²=0.575）。checkpoint选择bug已修复，
`best_corr_model.pth` 机制本身验证有效（本次重训确实按相关性正确保存了
epoch10而非val_loss更低的其他epoch）。后续如果重新训练（比如换用更大的
情绪数据集，或调整loss权重），这个机制会继续发挥作用。

### Step 2: MTG-Jamendo联合训练实验 (2026-07-12 凌晨，用户睡觉期间自主执行)

**假设**：DEAM单独训练Valence学不起来(R²≈0)是因为数据量/语义信息不够。
MTG-Jamendo mood/theme标签(59个，如happy/sad/dark/uplifting)带valence语义，
用它做辅助多标签分类任务、跟VA回归共享backbone联合训练，可能给Valence
"输血"。**这只是假设，训练结果如实记录，不做选择性汇报**。

数据管线：`data/mtg_jamendo.py`流式处理MTG音频分片(不整包落盘，处理完立刻
删原始音频，只留mel特征缓存)，累计缓存5786首(3倍于DEAM)。模型加了
`MoodHead`(`model/experts/mood_head.py`)，联合训练脚本`train/train_joint.py`
每step同时算DEAM的VA loss和MTG的mood BCE loss，加权(mood_weight=0.3)求和后
一次backward。从`best_model_run1_epoch4.pth`初始化backbone+emotion_head。

**Epoch 1-4 结果**（从`best_model_run1_epoch4.pth`初始化后）：

| Epoch | V_mean_R2 | V_mean_PearsonR | A_mean_R2 | A_mean_PearsonR | mood_micro_f1 |
|:-----:|:---------:|:---------------:|:---------:|:---------------:|:--------------:|
| 1 | -0.0588 | 0.2557 | 0.5134 | 0.7389 | 0.0000 |
| 2 | -0.0189 | 0.2808 | 0.5789 | 0.7627 | 0.0000 |
| 3 | -0.0146 | 0.2584 | 0.4175 | 0.7330 | 0.0000 |

Valence R²在朝着0的方向改善(-0.059→-0.019→-0.015)，但涨幅在收窄，还没转正，
不确定是真趋势还是训练噪声，需要更多epoch观察。Arousal保持良好(R² 0.42-0.58)
没被联合训练拖累。

**已知问题**：`mood_micro_f1`全程0——BCE loss在降(0.68→0.13)但模型在阈值0.5下
不产生任何命中，怀疑是多标签类别不均衡（59个标签里大多数track只有1-3个正
标签，模型可能学到"全预测为负"就能让loss很低）。BCE的梯度方向对所有类别
仍然是有信息量的（即使没有某个标签越过阈值，logit也在朝正确方向移动），
所以不确定这是否影响了它对共享表征的"辅助"作用——留待更多epoch数据判断。
如果这个指标持续为0，后续应该给`F.binary_cross_entropy_with_logits`加
`pos_weight`（按训练集算每个标签的负/正样本比重新加权）。

**训练稳定性教训**：一个epoch实测~13分钟(比最初估的45-50分钟快很多)，但
仍可能被这台机器上无法完全定位根因的后台任务终止机制打断（跟Step 1重训
时的现象一致）。给`train_joint.py`加了epoch内断点续训（每60step存
`checkpoints/joint_latest.pth`，`--resume`能从epoch内部继续，不需要重跑
整个epoch）。

**Epoch 4-5 更新（同一次运行，5786首数据集）**：V_mean_R2继续观察到
epoch4=-0.0552、epoch5=-0.1148——**不是单调改善，是震荡**，epoch3之后又
掉回比epoch1更差的水平。mood_micro_f1连续5个epoch都是0.0000，没有自己好转。

**诊断+修复**：直接查了训练集标签分布——平均每首歌只有1.8/59个标签为正
（中位数正例率2.3%，最不均衡的14个标签正例率<1%）。这解释了mood_micro_f1
卡在0的原因：不加权的BCE下，模型学"全预测为负"就能拿到很低loss，用sigmoid
阈值0.5衡量时永远不会有命中。**给`data/mtg_jamendo.py`加了
`compute_pos_weight()`**（按训练集算每个标签的负/正样本比，clip到30防止
极端稀有标签把训练搞不稳定），`train_joint.py`的BCE loss现在用这个
pos_weight。重新从`best_model_run1_epoch4.pth`干净重启（同样的基线checkpoint，
公平对比），日志见`logs_joint_v2.txt`。pos_weight统计：中位数和最大值都
顶到clip上限30（大多数标签正例率≤3.2%），min=9.3。mood_loss数值变大是
预期的（加权后不能跟之前直接比大小，要看F1/precision/recall）。

**pos_weight修复后 Epoch 1-3 结果**（`logs_joint_v2.txt`）：

| Epoch | V_mean_R2 | V_mean_PearsonR | A_mean_R2 | A_mean_PearsonR | mood_micro_f1 |
|:-----:|:---------:|:---------------:|:---------:|:---------------:|:--------------:|
| 1 | -0.0010 | 0.2584 | 0.4629 | 0.7214 | 0.1091 |
| 2 | -0.0361 | 0.2541 | 0.4506 | 0.7258 | 0.1023 |
| 3 | 0.0071 | 0.2886 | 0.4899 | 0.7389 | 0.1059 |

**mood_micro_f1终于脱离0**（0.10左右，说明模型真的学会预测部分标签了，
不再是"全预测为负"的退化解）。**V_mean_R2目前是这几次实验里表现最好的**——
一直贴着0线，epoch3甚至转正(0.0071)。**但要如实评估**：0.007的R²在实践
中约等于没有强信号，3个epoch的样本量也可能只是噪声，还不能说"假设被证实"，
只能说"目前唯一一次没有明显更差、且mood辅助任务本身确实学到东西的版本"。
继续观察更多epoch再做结论。

**Epoch 4-6 更新**：

| Epoch | V_mean_R2 | V_mean_PearsonR | A_mean_R2 | A_mean_PearsonR | mood_micro_f1 |
|:-----:|:---------:|:---------------:|:---------:|:---------------:|:--------------:|
| 4 | -0.0138 | 0.2829 | 0.4702 | 0.7281 | 0.1083 |
| 5 | -0.0797 | 0.2786 | 0.4177 | 0.7082 | 0.1118 |
| 6 | -0.0005 | 0.3179 | 0.4719 | 0.7273 | 0.1177 |

6个epoch的V_mean_R2完整序列：-0.001, -0.036, **+0.007**, -0.014, -0.080, -0.001
（均值≈-0.021）。**看起来是围绕一个接近0的小负值震荡，不是单调改善趋势**。
`mood_micro_f1`倒是在稳步涨（0.109→0.102→0.106→0.108→0.112→0.118），辅助
任务本身学得越来越好，但暂时没能转化成Valence的清晰提升。Arousal保持健康
(R2 0.42-0.49)。再跑几个epoch到8-10轮做阶段性总结。

### Step 2 阶段性总结 (9 epochs，诚实评估)

| Epoch | V_mean_R2 | V_mean_PearsonR | A_mean_R2 | A_mean_PearsonR | mood_micro_f1 |
|:-----:|:---------:|:---------------:|:---------:|:---------------:|:--------------:|
| 1 | -0.0010 | 0.2584 | 0.4629 | 0.7214 | 0.1091 |
| 2 | -0.0361 | 0.2541 | 0.4506 | 0.7258 | 0.1023 |
| 3 | 0.0071 | 0.2886 | 0.4899 | 0.7389 | 0.1059 |
| 4 | -0.0138 | 0.2829 | 0.4702 | 0.7281 | 0.1083 |
| 5 | -0.0797 | 0.2786 | 0.4177 | 0.7082 | 0.1118 |
| 6 | -0.0005 | 0.3179 | 0.4719 | 0.7273 | 0.1177 |
| 7 | 0.0276 | 0.3096 | 0.5261 | 0.7470 | 0.1253 |
| 8 | -0.0616 | 0.3385 | 0.5367 | 0.7516 | 0.1282 |
| 9 | 0.0358 | 0.3382 | 0.5763 | 0.7641 | 0.1310 |

**结论（不做选择性汇报，混合结果如实呈现）**：

1. **"MTG标签帮Valence学起来"这个假设，在R²层面没有得到确认**——9个epoch里
   V_mean_R2持续在-0.08到+0.036之间震荡，没有单调改善的证据，均值仍接近0
   甚至略偏负。跟纯DEAM训练的±0.02量级相比，没有本质区别。

2. **但V_mean_PearsonR有清楚、相当稳定的上升趋势**（0.258→0.338，9个点里
   除epoch5有小回调外基本单调），说明预测值和真实valence"方向"上的相关性
   确实在改善，只是R²（同时惩罚整体偏差/尺度失配，对相关性的度量比Pearson r
   更严格）没跟上。可能的解释：模型学到了valence的相对排序/趋势，但输出的
   绝对尺度/偏移还没校准好——这跟"完全没学到东西"是不同的失败模式，更接近
   "学到了但还不够精确"。

3. **意外之喜：Arousal持续变好，尚未触顶**。A_mean_R2从0.46一路涨到0.576
   （epoch9），**超过了本次实验的初始基线`best_model_run1_epoch4.pth`本身
   的0.575**——联合训练不仅没有拖累Arousal，反而可能因为额外的训练步数/
   MTG数据的共享表征正则化效果而进一步提升了它。

4. **mood_micro_f1稳步改善**（0.109→0.131），辅助任务本身学习状况良好，
   没有停滞。

**决策**：三个指标(mood_f1、Arousal、Valence-PearsonR)都还在改善、没有
触顶迹象，没有理由现在停止这个方向。继续训练观察V_mean_R2是否会跟随
PearsonR的趋势最终转为正向单调改善，或者维持震荡（如果一直不追上，说明
"校准问题"假设也需要修正）。

### Step 2 最终结论 (epoch 9 峰值，之后回落)

| Epoch | V_mean_R2 | V_mean_PearsonR | A_mean_R2 | A_mean_PearsonR | mood_micro_f1 |
|:-----:|:---------:|:---------------:|:---------:|:---------------:|:--------------:|
| 9 | 0.0358 | 0.3382 | **0.5763** | **0.7641** | 0.1310 |
| 10 | -0.0271 | 0.3401 | 0.5623 | 0.7503 | 0.1203 |
| 11 | -0.0812 | 0.3316 | 0.5494 | 0.7459 | 0.1336 |
| 12 | -0.0496 | 0.2902 | 0.4998 | 0.7312 | 0.1313 |

**Arousal 和 Valence-PearsonR 在 epoch 9 触顶后连续3个epoch同步回落**——
这不是单epoch噪声（噪声更可能像epoch5那样单点下探后立刻反弹），三个epoch
持续同向下降更像是真的过了拐点。`checkpoints/joint_best_corr_model.pth`
按设计正确捕获了这个峰值（epoch9, corr_score=0.5513），不会因为后续epoch
变差而丢失——checkpoint双轨机制（按val_loss和按相关性分别存）这次真正发挥
了作用。

**本阶段实验最终结论（如实汇报，不为了好看而美化）**：

1. **核心假设"MTG mood标签能帮Valence-R2学起来"，12个epoch的证据下没有
   被证实**。V_mean_R2全程在-0.08~+0.036震荡，均值接近0甚至略负，跟纯
   DEAM训练(±0.02量级)没有本质差异，也没有随训练推进而改善的趋势。
2. **但确实观察到两个真实的、值得记录的效果**：
   - Valence的Pearson相关系数从0.258一路涨到0.340(epoch10)才回落——峰值
     PearsonR比DEAM单独训练时的最好记录(0.30, epoch4_run1)更高，说明联合
     训练对"valence方向感"确有提升，只是没有转化成R²的改善（R²同时惩罚
     尺度/偏差失配，比Pearson r更严格）。
   - **Arousal意外获得了全实验最佳成绩**：R2=0.5763/PearsonR=0.7641
     (epoch9)，超过了本次实验初始基线`best_model_run1_epoch4.pth`自身的
     0.575/0.76。这可能是额外训练步数的功劳，也可能是MTG共享表征带来的
     正则化效果，本实验设计无法区分这两种解释。
3. **推荐产出**：`checkpoints/joint_best_corr_model.pth`(epoch9)是目前
   全部实验里综合最好的checkpoint——Arousal达到新高，Valence-PearsonR处于
   较高水平，mood分类也在正常工作。**建议作为当前阶段的"最终交付checkpoint"**，
   优于之前的`best_model_run1_epoch4.pth`和`best_corr_model.pth`(纯DEAM版)。
4. **对下一步的启示**：如果还想继续提升Valence-R2，"更多同类数据"这条路
   看起来收益有限（假设未被证实），更值得探索的方向可能是：
   - 针对Valence预测做显式的偏差/尺度校正（比如训练后对输出做线性重标定）
   - 引入真正携带语义的信号（歌词、和声进行等），而不是弱监督的mood标签
   - 这两个方向都还没有验证过，只是基于本次实验结果的合理猜测，不是结论。

### Step 2 收尾 (15 epochs，手动停止)

epoch9峰值之后，又观察了6个epoch(10-15)确认没有反弹：

| Epoch | V_mean_R2 | V_mean_PearsonR | A_mean_R2 | A_mean_PearsonR | mood_micro_f1 |
|:-----:|:---------:|:---------------:|:---------:|:---------------:|:--------------:|
| 10 | -0.0271 | 0.3401 | 0.5623 | 0.7503 | 0.1203 |
| 11 | -0.0812 | 0.3316 | 0.5494 | 0.7459 | 0.1336 |
| 12 | -0.0496 | 0.2902 | 0.4998 | 0.7312 | 0.1313 |
| 13 | -0.0675 | 0.3208 | 0.5534 | 0.7490 | 0.1434 |
| 14 | -0.1271 | 0.3072 | 0.5280 | 0.7398 | 0.1401 |
| 15 | -0.0634 | 0.3269 | 0.5215 | 0.7348 | 0.1380 |

val_loss从epoch7(0.3953)之后再没改善过，到epoch15已经8轮未破纪录（早停
耐心15，本可以再等7轮到epoch22自然早停，但6个epoch的post-peak观察已经
足够确认没有反弹——Arousal稳定在0.73-0.75的PearsonR区间，比峰值0.764略低
但不再持续下滑；mood_f1还在小幅波动上涨(0.12-0.14)但对Valence-R2没有
进一步帮助）。**手动停止训练**，不等剩余7轮的早停耐心耗尽，避免占用GPU
时间换取已经能预判的结果。

**最终交付**：`checkpoints/joint_best_corr_model.pth` (epoch9)
- Arousal: R2=0.5763, PearsonR=0.7641, CCC=0.7168 —— **全部实验里最好的
  Arousal成绩**，超过最初基线的0.575
- Valence: R2=0.0358, PearsonR=0.3382, CCC=0.2892 —— R2依然接近0(没有
  实用意义上的强信号)，但PearsonR是所有实验里最高的
- mood分类: micro_f1=0.131 —— 辅助任务学习状况正常

**核心假设的最终判定**："MTG mood/theme标签联合训练能让Valence-R2学起来"
**未获证实**。改善主要体现在Arousal(可能是训练量增加+MTG共享表征的副产物，
非本意但真实)和Valence的相关系数(而非R2)。如果用户后续想继续攻克Valence，
基于本次结果，"堆更多同类弱监督数据"预期收益有限，值得先尝试的是输出端的
偏差/尺度校正，或者干脆接受"纯音频信号对Valence的预测力有限"这个音乐情感
识别领域的已知天花板，把精力转向乐器/体裁/节拍检测等其他更容易出效果的
维度。

### Step 3: 输出端校准诊断 (验证"没学到"还是"没校准好")

用`train/calibration_check.py`按歌曲切成两半(不能同集fit+测，否则R²数学上
必然等于PearsonR²，是循环论证)：一半拟合线性校准`true≈a*pred+b`，另一半上
测校准前后的R²。

**结果**：eval集上未校准R²=-0.2279，校准后R²=+0.0211（改善+0.249）。预测值
标准差(0.129)明显小于真实值标准差(0.212)——模型的valence输出"收缩"向均值，
不敢给出极端预测，这是典型的"学到了方向但幅度没校准"的信号。

**结论（两层都要看）**：
1. 校准确实有效——支持"部分是尺度/偏差问题"的猜测，这是一个真实、廉价、
   值得保留的改进。
2. 但校准后R²依然很小(0.02)——**校准不是银弹**，模型对valence的底层信息
   量本身还是弱的(r²≈0.11量级的天花板)，光调整输出的尺度/偏移解决不了
   根本的信息不足问题。

**已落地**：`checkpoints/joint_best_corr_model.calibration.json`——在DEAM
val全集(267首)上拟合的最终校准系数(valence: a=0.392,b=-0.002；arousal:
a=0.712,b=0.013，注意arousal也有轻微欠校准，只是没valence那么严重)。推理
时可以对emotion_head原始输出做这个线性后处理，不需要改模型权重。

**新增 `infer_emotion.py`**：把checkpoint+校准串成能直接用的推理工具——
输入任意长度音频，滑窗(15秒窗/5秒步长)扫过全曲，输出随时间变化的
Valence/Arousal曲线(JSON+可选绘图)。这个时间序列以后可以直接接入
`beatmap/chart_generator.py`做"情绪驱动note密度"。踩了一个坑：一开始复用
了`utils/audio.py`的`load_audio(max_duration=...)`，以为`max_duration`只是
个上限保护，实际它会把音频**强制pad到那个长度**——传了3600秒结果把45秒的
DEAM测试曲目填充成了1小时静音，滑窗白扫了一大堆零信号。已修复：改用不做
任何pad的独立加载函数，`max_duration`语义改成"仅截断超长音频，从不填充
短音频"。

### Step 4: 歌词增强路线调研 (结论：不值得投入)

用户提议加歌词信号帮Valence突破瓶颈。调研+实测两步走：

1. **DEAM确实有官方metadata**（`metadata.zip`，之前迁移时没下，已补充到
   `data/deam_meta/`）：`metadata_2013/2014/2015.csv`三个文件合并后
   **100%覆盖全部1802首**曲目的artist/title/genre。
2. **Genre分布比预期乐观**：Country/Rock/Pop/Folk/Blues/Hip-Hop/SoulRB
   (通常有人声)占57.6%(1037/1802)，DEAM像是刻意做了接近GTZAN的10类genre
   平衡采样，不是纯粹冷门器乐向的FMA原始分布——**但这个乐观推论被实测
   否定了**，见下。
3. **实测**：从"vocal-likely genre"里随机抽150首，用lrclib.net(免费/无
   需API key/不违反ToS，Genius API拿不到真歌词只有搜索结果，爬网页违反
   ToS已排除)查真实歌词匹配率。**结果：21/150 = 14.0%**，落在调研阶段
   估计的10-30%区间低端。
4. **看明细找根因**：几乎所有miss都是真正意义上的冷门独立/Bandcamp/FMA
   原生创作者(如"Jason Shaw"、"Mountain Cult"、"The Peach Tree"等)——
   genre标签为"有人声的类型"不代表这个具体艺人真的有被任何数据库收录过
   的歌词，DEAM的艺人池整体上极度小众，不是"genre分布好看就该有歌词"这个
   推论能覆盖的。命中的几首都是有一定知名度的另类/独立乐队(Nine Inch
   Nails, Jonathan Coulton, Le Tigre, Daniel Johnston等)。

**最终结论：歌词增强路线不值得投入**。14%匹配率意味着1780首里最多能配上
词的也就200首左右，这个规模不足以给一个本来就信号稀薄的Valence回归任务
提供有意义的额外训练信号，反而会引入"大部分样本没有歌词特征"的稀疏性
问题。**不建议继续往这个方向投入工程量**，除非未来换一个歌词覆盖率本身
更高的数据集(比如直接用主流流行乐/知名歌手作为训练数据源，但那样就要
放弃DEAM现成的VA标注，是另一个量级的工作)。

### Step 5: 真实网络音乐 out-of-distribution benchmark (2026-07-12)

之前所有评测(Step 1-3)都在DEAM val集上，本质是"同分布"评测。这次找了
incompetech.com(Kevin MacLeod的CC-BY授权免版税曲库)的4首风格迥异的曲子
做**分布外**测试——这是模型第一次跑在完全没见过的真实音乐上。受限于本机
网络带宽极慢(实测约8-20KB/s，连google.com都会超时，怀疑是sandbox网络
限速)，用HTTP Range请求只下了每首的前19-34秒片段，够跑1-2个15秒滑窗即可，
不影响情绪推理/踩谱pipeline的有效性验证。

脚本：`train/benchmark_realworld.py`（一次性加载模型，对`joint_best_corr_model.pth`
+校准文件跑`infer_emotion.analyze_emotion_curve` + `beatmap/chart_generator.generate_chart`，
逐曲打印+落盘`docs/benchmark_realworld/results.json`和4张曲线图）。

| 曲目 | 预期基调 | Valence(mean) | Arousal(mean) | 方向判定 |
|---|---|---|---|---|
| Carefree (轻快民谣) | V正/A中 | +0.063 | +0.042 | V✓ |
| Killers (黑暗/紧张) | V负/A高 | +0.030 | +0.254 | V✗ A✓ |
| Monkeys Spinning Monkeys (滑稽/俏皮) | V正/A高 | +0.048 | +0.034 | V✗ A✗ |
| Wallpaper (环境背景) | V中性/A低 | +0.073 | +0.096 | A✗ |

**方向判定 2/6，跟瞎猜差不多。诚实结论：**

1. **Valence在分布外基本没有泛化。** 4首风格完全不同的曲子，Valence预测
   全部落在 [+0.030, +0.073] 这个极窄区间里，几乎是个跟输入无关的常数(接近
   校准截距)。这跟Step 2/3已经发现的"DEAM val上Valence R²从未转正、只能靠
   线性校准把R²从-0.23拉到+0.02"是一致的——不是校准诊断的巧合，是Valence
   这个头本身没学到能跨分布迁移的东西，DEAM同分布内的表现本来就贴近噪声
   下限。
2. **Arousal部分泛化，但不可靠。** 正确把Killers(预期最高)排到4首里最高，
   跟DEAM val上Arousal是全实验最佳(R²=0.576)这个结论方向一致；但Wallpaper
   (预期最低，纯氛围乐)反而被排到第二高，说明"分布外排序基本靠谱"这个
   结论撑不住，只能说"极端情况(强烈紧张感)大概率能抓住，细粒度排序不能信"。
3. **这不是一个新问题，是已知问题的复现验证。** 之前的怀疑一直停留在
   DEAM val集内部的统计推断(R²、校准残差)，这次是第一次拿真实分布外音频
   验证，结果跟怀疑吻合——如果之前有任何"也许DEAM val的负R²只是拟合校准
   参数的问题、模型本身可能还不错"的一线希望，这次可以排除了。

**速度benchmark（附带的好消息）**：GTX 1650上情绪推理RTF≈0.01(处理1秒音频
平均10ms，比实时快约100倍)，一首完整3-4分钟歌曲跑完情绪曲线预计2-4秒。
踩谱生成(`chart_generator`)单曲通常<0.2秒，唯一例外是进程内第一次调用耗时
10.4秒——这是librosa内部numba JIT编译的一次性开销，不是每首歌都要付，不算
真实的每曲成本。**结论：速度完全不是问题，如果要投入工程量，应该投在Valence
这个头本身(比如换更好的预训练特征/更大规模的强监督数据)而不是推理性能。**

原始数据：`docs/benchmark_realworld/results.json`；4张曲线图同目录下
`*_curve.png`。

### 下一步（用户已澄清项目终极目标）

项目目标是音游自动写谱工具（自动情感分析+自动踩音），不是纯研究。情绪回归
只解决"氛围/难度曲线风格"这一层；"自动踩音"真正需要的是**节拍/onset检测**
（毫秒级精度），这是与情绪回归完全不同的技术路线，当前架构未覆盖。建议下一
阶段优先做这个，可以先用 librosa 的 `onset_detect`/`beat_track` 做传统信号
处理基线（不需要训练，立等可用），验证够不够用后再决定是否需要专门训练的
节拍跟踪模型。
