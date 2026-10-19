# MusicLEARN (musiqueer)

> **本项目已暂停维护并归档。** 代码保留作参考。

多任务音乐分析模型：输入一段音频，输出**情绪、流派、乐器、风格/氛围标签**等分析结果。内部代号 MusicUnderstand。

## 结构

| 目录 / 文件 | 内容 |
|---|---|
| `model/` | 主干网络 `backbone.py`，多任务模型 `music_understand.py`，各任务头 `experts/`（emotion / genre / instrument / mood / language_decoder 等） |
| `data/` | 数据集加载：DEAM、FMA、MTG-Jamendo、MusicCaps、Music4All 等（多为流式读取），以及 DSP 伪标签 |
| `train/` | 各任务与联合训练脚本（`train_multitask.py`、`train_unified.py`、`train_v3.1.py` 等）和评测脚本 |
| `utils/` | 梅尔频谱等音频处理、指标 |
| `config/default.yaml` | 模型与训练配置 |
| `infer.py` | 单曲推理：20 类流派、20 种乐器（OpenMIC）、59 个 MTG 情绪标签 |
| `autotag.py` | 批量分析工具，支持单文件 / 目录、`--json`、`--plot` 情绪曲线 |
| `benchmark/` | 8 首测试曲目的分析报告（`BENCHMARK.md`） |
| `docs/` | 进度、路线图、技术改进记录 |

## 运行

```bash
pip install torch torchaudio librosa tokenizers pyyaml tqdm pandas numpy
python infer.py 歌曲.mp3
python autotag.py 音乐文件夹/ --json
```

注意：仓库**不包含训练好的权重**（`checkpoints/*.pth` 已在 `.gitignore` 中排除），`infer.py` 默认读取 `checkpoints/unified.pth`，需要自行训练或另行获得权重后才能推理。数据集缓存同样未入库。