# MusicUnderstand 3.0 — 技术改进方案

## 1. Cross-Attention 增强：让 Decoder 真正"听到"音频

**问题**: Decoder 的 cross-attention 没被强制使用。本地跑纯文本训练时用了噪声替代，模型学会了忽略音频。

**方案**: 加一个 **Attention Gate**，在 cross-attention 输出和 decoder 输出之间加门控，门控值由音频特征的能量决定。音频没信息时门控关闭，decoder 当纯语言模型用；音频有信息时门控打开。

```python
class AudioGatedCrossAttention(nn.Module):
    """音频门控交叉注意力"""
    def forward(self, x, memory):
        # 标准 cross-attention
        attn_out = self.cross_attn(x, memory, memory)
        # 音频能量门控: 从 memory 算一个标量 gate
        gate = torch.sigmoid(self.gate_proj(memory.mean(dim=1)))
        # 门控后的输出
        return x + gate.unsqueeze(1) * attn_out
```

## 2. Scheduled Sampling：让模型学会自己走路

**问题**: 训练时给 ground-truth token (teacher forcing)，推理时用自己的输出。训练和推理的输入分布不一样，一步错步步错。

**方案**: 训练时以概率 p 用模型自己的输出代替 ground truth。p 从 1.0 逐渐退火到 0.0。

```python
# 训练循环里
for epoch in range(num_epochs):
    sampling_prob = max(0.0, 1.0 - epoch / num_epochs * 0.8)
    # 对每个 token，以 sampling_prob 的概率用模型自己的预测
    if random.random() < sampling_prob and step > 0:
        # 用上一个位置的预测作为输入
        input_ids[:, step] = predicted_ids[:, step-1]
```

## 3. 多尺度特征融合：同时看到森林和树木

**问题**: Backbone 只输出最后一层 256 维特征。音乐既有整体情绪又有局部变化。

**方案**: 把 backbone 各层的特征都拿出来，层次化融合。

```python
class MultiScaleBackbone(nn.Module):
    """多尺度 Backbone"""
    def forward(self, mel):
        outputs = []
        x = self.patch_embed(mel)
        x = self.pos_encoder(x)
        for block in self.blocks:
            x = block(x)
            outputs.append(x)  # 每层都存
        # 融合浅层(局部)和深层(全局)特征
        local_feat = outputs[0]  # 浅层细节
        global_feat = outputs[-1]  # 高层语义
        return torch.cat([local_feat, global_feat], dim=-1)
```

## 4. Contrastive Audio-Text 对齐

**问题**: 不同任务的 head 各自为政，backbone 不知道学什么特征对描述有用。

**方案**: 在训练 Decoder 的同时，加一个对比学习分支。把同一首歌的音频特征和文本描述拉近，不同歌的推远。
