"""
MusicUnderstand 3.0 — 主模型 (含 Language Decoder)

组装 backbone + 各 Expert Head。
Step 1:  emotion head
Step 3 (当前): + language decoder (caption generation)

设计特点:
  - active_experts 控制推理时启用哪些 head
  - 每个 head 独立的前向传播路径
  - LanguageDecoder: 属于"多模态"head，需要文本 token 输入
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any, Union

from .backbone import MusicTransformerBackbone
from .experts.emotion_head import EmotionHead
from .experts.language_decoder import LanguageDecoder
from .experts.genre_head import GenreHead
from .experts.instrument_head import InstrumentHead
from .experts.mood_head import MoodHead


class MusicUnderstand2(nn.Module):
    """
    MusicUnderstand 3.0 主模型

    Architecture:
      Mel [B, 1, 128, T]
        → TransformerBackbone [B, T, 256]
        ├→ EmotionHead [B, T, 12]
        ├→ GenreHead [B, num_genres]
        ├→ InstrumentHead [B, num_instruments]  (Sigmoid)
        └→ LanguageDecoder [B, T_text, vocab_size]  (with cross-attention to backbone)
    """

    def __init__(
        self,
        backbone_config: Optional[Dict[str, Any]] = None,
        emotion_config: Optional[Dict[str, Any]] = None,
        language_config: Optional[Dict[str, Any]] = None,
        genre_config: Optional[Dict[str, Any]] = None,
        instrument_config: Optional[Dict[str, Any]] = None,
        mood_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        if backbone_config is None:
            backbone_config = {}
        if emotion_config is None:
            emotion_config = {}
        if genre_config is None:
            genre_config = {}
        if instrument_config is None:
            instrument_config = {}
        if mood_config is None:
            mood_config = {}

        self.backbone = MusicTransformerBackbone(**backbone_config)

        # Emotion Head (always available)
        self.emotion_head = EmotionHead(**emotion_config)

        # Genre Head (音乐体裁分类, 可选)
        self.genre_head = GenreHead(**genre_config)

        # Instrument Head (乐器检测, 可选)
        self.instrument_head = InstrumentHead(**instrument_config)

        # Mood Head (MTG-Jamendo mood/theme 多标签, 辅助任务用于给emotion_head
        # 的valence回归"输血" — 见 model/experts/mood_head.py 顶部注释)
        self.mood_head = MoodHead(**mood_config)

        # Language Decoder (optional, set up if config provided)
        if language_config is not None:
            self.language_decoder = LanguageDecoder(**language_config)
            self.has_language_decoder = True
        else:
            self.language_decoder = None
            self.has_language_decoder = False

    def forward(
        self,
        mel: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        active_experts: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            mel: [B, 1, n_mels, T] Mel频谱
            input_ids: [B, T_text] 文本 tokens (仅在使用 language_decoder 时需要)
            attention_mask: [B, T_text] attention mask (仅在使用 language_decoder 时需要)
            active_experts: 要启用的head列表
                e.g. ['emotion', 'language', 'genre', 'instrument']

        Returns:
            outputs: dict {
                'emotion': [B, T, 12]             情绪输出（可选）
                'genre_logits': [B, num_genres]   体裁 logits（可选）
                'instrument_logits': [B, num_inst] 乐器 logits（可选）
                'language_logits': [B, T, V]     语言模型 logits（可选）
                'memory': [B, T, d_model]          backbone 输出（可选）
            }
        """
        if active_experts is None:
            active_experts = ['emotion']
            if self.has_language_decoder:
                active_experts.append('language')

        x = self.backbone(mel)  # [B, T, d_model]

        outputs = {}

        if 'emotion' in active_experts:
            outputs['emotion'] = self.emotion_head(x)  # [B, T, 12]

        if 'genre' in active_experts:
            outputs['genre_logits'] = self.genre_head(x)  # [B, num_genres]

        if 'instrument' in active_experts:
            outputs['instrument_logits'] = self.instrument_head(x)  # [B, num_instruments]

        if 'mood' in active_experts:
            outputs['mood_logits'] = self.mood_head(x)  # [B, num_mood_tags]

        if 'language' in active_experts and self.has_language_decoder:
            assert input_ids is not None, (
                "input_ids required for language decoder"
            )
            language_logits = self.language_decoder(
                memory=x,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            outputs['language_logits'] = language_logits  # [B, T, V]

        outputs['memory'] = x  # Expose backbone output

        return outputs

    @torch.no_grad()
    def generate_caption(
        self,
        mel: torch.Tensor,
        tokenizer: 'AutoTokenizer',
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> List[str]:
        """
        生成音乐描述（便捷接口）

        Args:
            mel: [B, 1, n_mels, T] Mel频谱
            tokenizer: GPT-2 tokenizer
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_k: top-k 采样

        Returns:
            captions: 文本描述列表
        """
        if not self.has_language_decoder:
            raise RuntimeError("Language decoder not initialized")

        self.eval()
        memory = self.backbone(mel)  # [B, T, d_model]
        return self.language_decoder.generate(
            memory=memory,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    @torch.no_grad()
    def inference(self, mel: torch.Tensor,
                  active_experts: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        推理接口（不启用梯度）

        Args:
            mel: [B, 1, n_mels, T]
            active_experts: 要启用的head列表
                default: ['emotion']

        Returns:
            所有 head 的输出
        """
        if active_experts is None:
            active_experts = ['emotion']
        self.eval()
        return self.forward(mel, active_experts=active_experts)


# ============================================================
# 单元测试
# ============================================================
if __name__ == "__main__":
    B, T = 4, 60
    dummy_mel = torch.randn(B, 1, 128, T)

    # Test without language decoder (backward compatible)
    print("=" * 50)
    print("Test 1: Without language decoder (backward compatible)")
    model = MusicUnderstand2()
    output = model(dummy_mel, active_experts=['emotion'])

    total_params = sum(p.numel() for p in model.parameters())
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    head_params = sum(p.numel() for p in model.emotion_head.parameters())

    print(f"  Emotion output: {output['emotion'].shape}")
    print(f"  总参数: {total_params:,}")

    # Test with language decoder
    print("\n" + "=" * 50)
    print("Test 2: With language decoder")
    model_lang = MusicUnderstand2(
        language_config={
            "d_model": 256,
            "nhead": 4,
            "num_layers": 2,
            "vocab_size": 1000,  # Small for testing
        }
    )

    dummy_tokens = torch.randint(0, 1000, (B, 32))
    dummy_mask = torch.ones((B, 32), dtype=torch.long)
    output_lang = model_lang(
        dummy_mel,
        input_ids=dummy_tokens,
        attention_mask=dummy_mask,
        active_experts=['emotion', 'language'],
    )

    print(f"  Emotion:     {output_lang['emotion'].shape}")
    print(f"  Language:    {output_lang['language_logits'].shape}")
    print(f"  Memory:      {output_lang['memory'].shape}")
    total_lang = sum(p.numel() for p in model_lang.parameters())
    print(f"  总参数 (含 decoder): {total_lang:,}")

    # Test with genre and instrument heads
    print("\n" + "=" * 50)
    print("Test 3: With genre and instrument heads")
    output_all = model(
        dummy_mel,
        active_experts=['emotion', 'genre', 'instrument'],
    )

    print(f"  Emotion:         {output_all['emotion'].shape}")
    print(f"  Genre logits:    {output_all['genre_logits'].shape}")
    print(f"  Instrument logits: {output_all['instrument_logits'].shape}")
    print(f"  Memory:          {output_all['memory'].shape}")
    total_all = sum(p.numel() for p in model.parameters())
    print(f"  总参数 (含 genre + instrument): {total_all:,}")

    print("\n[OK] MusicUnderstand 3.0 构建成功")
