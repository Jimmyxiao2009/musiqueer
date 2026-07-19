"""
MusicUnderstand 4.0 â€” Differentiable End-to-End ReAct Decoder

ä»ç¦»æ•£å­—ç¬¦ä¸²å·¥å…·è°ƒç”¨ å‡çº§ä¸º è¿ç»­å¯å¾®ä¸“å®¶èåˆã€‚

æ ¸å¿ƒæ”¹åŠ¨:
  1. SoftRouter: ä» decoder hidden state è®¡ç®—ä¸“å®¶æ··åˆæƒé‡
  2. æ‰€æœ‰å·¥å…·è¿”å›è¿ç»­ç‰¹å¾å‘é‡è€Œéå­—ç¬¦ä¸²
  3. Weighted Feature Injection: åŠ æƒèåˆä¸“å®¶ç‰¹å¾ â†’ æ³¨å…¥ decoder hidden
  4. ç§»é™¤ @torch.no_grad(), numpy, å­—ç¬¦ä¸² tokenize æ–­ç‚¹
  5. æ”¯æŒè®­ç»ƒ (teacher forcing) å’Œæ¨ç† (è‡ªå›å½’)

æ¶æ„:
  User: "Analyze this music..."
    â†’ Token Embedding + Pos Encoding
    â†’ [Decoder Layer Ã— N]
        1. Self-Attention
        2. SoftRouter(decoder_hidden) â†’ [Î±â‚, Î±â‚‚, Î±â‚ƒ, Î±â‚„]
        3. All 4 experts forward from memory
        4. Î±-weighted fusion â†’ project â†’ add to hidden
        5. Cross-Attention(memory)
        6. FFN
    â†’ LM Head â†’ logits

æ¢¯åº¦è·¯å¾„:
  text_loss. backward()
    â†’ LM Head â† Decoder Layer â† Î±-weighted expert features
    â†’ SoftRouter (æ¢¯åº¦é€šè¿‡ Î± æƒé‡å›ä¼ )
    â†’ EmotionHead / GenreHead / InstrumentHead / TempoEstimator
    â†’ Backbone (å¯é€‰å†»ç»“)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any, Tuple, Union


# ============================================================
# SoftRouter â€” è¿ç»­å¯å¾®å·¥å…·é€‰æ‹©å™¨
# ============================================================

class SoftRouter(nn.Module):
    """
    ä» decoder hidden state è®¡ç®—ä¸“å®¶æ··åˆæƒé‡ã€‚

    è¾“å…¥: decoder hidden [B, T, d_model]
    è¾“å‡º: [B, num_experts] softmax æƒé‡

    æ¯ä¸ª token ä½ç½®å¯ä»¥æœ‰ä¸åŒçš„ä¸“å®¶æƒé‡ï¼Œä½†ä¸ºäº†ç¨³å®šæ€§ï¼Œ
    æˆ‘ä»¬å–åºåˆ—å¹³å‡åè·¯ç”±ã€‚
    """

    def __init__(self, d_model: int, num_experts: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, decoder_hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            decoder_hidden: [B, T, d_model] â€” æœ€åä¸€å±‚ DecoderLayer çš„è¾“å‡º

        Returns:
            weights: [B, num_experts] â€” softmax æƒé‡
        """
        pooled = decoder_hidden.mean(dim=1)  # [B, d_model]
        logits = self.router(pooled)         # [B, num_experts]
        weights = F.softmax(logits, dim=-1)  # [B, num_experts]
        return weights


# ============================================================
# å¯å¾®å·¥å…·: æ¯ä¸ª expert head åŒ…è£…æˆè¿ç»­ç‰¹å¾è¾“å‡º
# ============================================================

class DiffEmotionTool(nn.Module):
    """
    å¯å¾®æƒ…ç»ªåˆ†æå·¥å…·ã€‚
    è¾“å…¥: memory [B, T, d_model]
    è¾“å‡º: [B, 4] â€” valence_mean, arousal_mean, v_trend, a_trend

    æ³¨æ„: å¯¹æ—¶åºç»´åº¦åšå¹³å‡æ± åŒ–ï¼Œç¡®ä¿è¾“å‡ºæ’ä¸º [B, 4] è€Œé [B, T, 4]ã€‚
    """

    def __init__(self, emotion_head: nn.Module):
        super().__init__()
        self.head = emotion_head

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        """
        memory: [B, T, d_model]
        returns: [B, 4] â€” [valence_mean, arousal_mean, v_trend, a_trend]
        """
        out = self.head(memory)                         # [B, T, 12] or [B, 12]
        # ç¡®ä¿åšæ—¶é—´ç»´æ± åŒ–ï¼ˆå¤´å¯èƒ½å·²ç»åšäº†ï¼Œä¹Ÿå¯èƒ½æ²¡åšï¼‰
        if out.dim() == 3:
            v_mean = out[:, :, 0].mean(dim=1, keepdim=True)  # [B, 1]
            a_mean = out[:, :, 3].mean(dim=1, keepdim=True)  # [B, 1]
            v_start = out[:, 0:1, 0]                         # [B, 1]
            v_end = out[:, -1:, 0]                           # [B, 1]
            a_start = out[:, 0:1, 3]
            a_end = out[:, -1:, 3]
        else:
            v_mean = out[:, 0:1]     # [B, 1]
            a_mean = out[:, 3:4]
            v_start = v_mean         # æ²¡æœ‰æ—¶åºï¼Œç”¨å…¨å±€å€¼
            v_end = v_mean
            a_start = a_mean
            a_end = a_mean
        v_trend = v_end - v_start                        # [B, 1]
        a_trend = a_end - a_start
        return torch.cat([v_mean, a_mean, v_trend, a_trend], dim=-1)  # [B, 4]


class DiffGenreTool(nn.Module):
    """
    å¯å¾®ä½“è£åˆ†æå·¥å…·ã€‚
    è¾“å…¥: memory [B, T, d_model]
    è¾“å‡º: [B, num_genres] â€” softmax over genres
    """

    GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
              'jazz', 'metal', 'pop', 'reggae', 'rock']
    NUM_GENRES = len(GENRES)

    def __init__(self, genre_head: nn.Module):
        super().__init__()
        self.head = genre_head

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        """
        memory: [B, T, d_model]
        returns: [B, NUM_GENRES] â€” genre logits

        ä¿è¯: å§‹ç»ˆè¿”å› [B, num_genres]ï¼Œå³ä½¿ head æœ¬èº«æ²¡æœ‰åšæ—¶é—´æ± åŒ–ã€‚
        """
        logits = self.head(memory)  # [B, T, NUM_GENRES] or [B, NUM_GENRES]
        if logits.dim() == 3:
            logits = logits.mean(dim=1)  # [B, NUM_GENRES]
        return logits


class DiffInstrumentTool(nn.Module):
    """
    å¯å¾®ä¹å™¨æ£€æµ‹å·¥å…·ã€‚
    è¾“å…¥: memory [B, T, d_model]
    è¾“å‡º: [B, num_instruments] â€” ä¹å™¨ logits (ç”¨äº BCEWithLogitsLoss)

    ä¿è¯: å§‹ç»ˆè¿”å› [B, num_instruments]ã€‚
    """

    INSTRUMENTS = ['piano', 'drums', 'bass', 'guitar', 'violin',
                   'cello', 'flute', 'trumpet', 'saxophone', 'vocals']
    NUM_INSTRUMENTS = len(INSTRUMENTS)

    def __init__(self, instrument_head: nn.Module):
        super().__init__()
        self.head = instrument_head

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        """
        memory: [B, T, d_model]
        returns: [B, NUM_INSTRUMENTS] â€” logits
        """
        logits = self.head(memory)  # [B, T, NUM_INSTRUMENTS] or [B, NUM_INSTRUMENTS]
        if logits.dim() == 3:
            logits = logits.mean(dim=1)  # [B, NUM_INSTRUMENTS]
        return logits


class DiffTempoTool(nn.Module):
    """
    å¯å¾®èŠ‚å¥åˆ†æå·¥å…·ï¼ˆçº¯ torchï¼Œæ—  numpyï¼‰ã€‚

    ä» memory çš„èƒ½é‡å˜åŒ–ä¼°ç®— BPM å’ŒèŠ‚å¥ç‰¹å¾ã€‚
    ä½¿ç”¨ STFT é£æ ¼çš„å¸§çº§èƒ½é‡åˆ†æï¼Œå®Œå…¨ä¿æŒè®¡ç®—å›¾ã€‚
    """

    def __init__(self, expected_frame_rate: float = 2.0):  # 2 frames/sec (0.5s/å¸§)
        super().__init__()
        self.frame_rate = expected_frame_rate

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        """
        memory: [B, T, d_model]
        returns: [B, 3] â€” [energy_mean, energy_std, rhythmic_energy]

        æ‰€æœ‰æ“ä½œéƒ½åœ¨ torch å†…å®Œæˆï¼Œæ¢¯åº¦å®Œæ•´ã€‚
        """
        energy = memory.norm(dim=-1)                          # [B, T]

        energy_mean = energy.mean(dim=-1, keepdim=True)       # [B, 1]
        energy_std = energy.std(dim=-1, keepdim=True)         # [B, 1]

        # èŠ‚å¥èƒ½é‡: è®¡ç®—å¸§é—´èƒ½é‡å˜åŒ–çš„å¹…åº¦
        diffs = energy[:, 1:] - energy[:, :-1]                # [B, T-1]
        rhythmic_energy = diffs.abs().mean(dim=-1, keepdim=True)  # [B, 1]

        return torch.cat([energy_mean, energy_std, rhythmic_energy], dim=-1)  # [B, 3]


# ============================================================
# ExpertFeatureFuser â€” ä¸“å®¶ç‰¹å¾èåˆæ¨¡å—
# ============================================================

class ExpertFeatureFuser(nn.Module):
    """
    å°†å¤šä¸ªä¸“å®¶ç‰¹å¾åŠ æƒèåˆå¹¶æŠ•å½±å› d_modelã€‚

    æ¯ä¸ª expert è¾“å‡ºä¸åŒç»´åº¦ç‰¹å¾ï¼Œåˆ†åˆ«æŠ•å½±åˆ° d_model åï¼Œ
    æŒ‰ router æƒé‡åŠ æƒæ±‚å’Œã€‚
    """

    def __init__(self, d_model: int, num_experts: int = 4):
        super().__init__()
        # å„ expert çš„ç‰¹å¾ç»´åº¦:
        # emotion: 4, genre: NUM_GENRES(10), instrument: NUM_INSTRUMENTS(10), tempo: 3
        self.emotion_proj = nn.Linear(4, d_model)
        self.genre_proj = nn.Linear(DiffGenreTool.NUM_GENRES, d_model)
        self.instrument_proj = nn.Linear(DiffInstrumentTool.NUM_INSTRUMENTS, d_model)
        self.tempo_proj = nn.Linear(3, d_model)

        self.gate = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        weights: torch.Tensor,           # [B, 4]
        emotion_feat: torch.Tensor,       # [B, 4]
        genre_feat: torch.Tensor,         # [B, 10]
        instrument_feat: torch.Tensor,    # [B, 10]
        tempo_feat: torch.Tensor,         # [B, 3]
        decoder_hidden: torch.Tensor,     # [B, T, d_model]
    ) -> torch.Tensor:
        """
        Returns:
            fused: [B, T, d_model] â€” æ·»åŠ åˆ° decoder hidden çš„æ®‹å·®
        """
        # æŠ•å½±å„ expert ç‰¹å¾åˆ° d_model
        B = weights.size(0)
        emo_d = self.emotion_proj(emotion_feat)        # [B, d_model]
        gen_d = self.genre_proj(genre_feat)            # [B, d_model]
        ins_d = self.instrument_proj(instrument_feat)  # [B, d_model]
        tmp_d = self.tempo_proj(tempo_feat)            # [B, d_model]

        # åŠ æƒèåˆ: [B, d_model]
        expert_stack = torch.stack([emo_d, gen_d, ins_d, tmp_d], dim=1)  # [B, 4, d_model]
        weights_3d = weights.unsqueeze(-1)                               # [B, 4, 1]
        fused = (expert_stack * weights_3d).sum(dim=1)                   # [B, d_model]

        # é—¨æ§: å…è®¸ decoder æ§åˆ¶ expert æ³¨å…¥é‡
        gate_val = self.gate(decoder_hidden.mean(dim=1))                 # [B, 1]

        # [B, 1] * [B, 1, d_model] â†’ [B, 1, d_model]
        # gate_val éœ€è¦ unsqueeze åˆ° [B, 1, 1] é˜²æ­¢é”™è¯¯å¹¿æ’­
        return gate_val.unsqueeze(-1) * fused.unsqueeze(1)               # [B, 1, d_model]


# ============================================================
# ä¿®æ”¹åçš„ DecoderLayer â€” åµŒå…¥ Expert Feature Injection
# ============================================================

class DiffDecoderLayer(nn.Module):
    """
    æ”¯æŒå¯å¾®ä¸“å®¶ç‰¹å¾æ³¨å…¥çš„ Decoder Layerã€‚

    Pre-LN ç»“æ„ + Expert Feature Injectionï¼š
      1. Self-Attention
      2. Expert Feature Injection (router â†’ expert fusion)
      3. Cross-Attention (åˆ° memory)
      4. FFN
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_experts: int = 4,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.audio_gate = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())

        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        self.norm_expert = nn.LayerNorm(d_model)  # Expert injection åçš„ Norm

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        expert_residual: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tgt: [B, T, d_model] â€” decoder è¾“å…¥
            memory: [B, T_audio, d_model] â€” backbone è¾“å‡º
            expert_residual: [B, 1, d_model] â€” å¯å¾®ä¸“å®¶ç‰¹å¾æ³¨å…¥ï¼ˆå¯é€‰ï¼‰
        """
        # Pre-LN Self-Attention
        residual = tgt
        x = self.norm1(tgt)
        x = self.self_attn(x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask, need_weights=False)[0]
        x = residual + x

        # Expert Feature Injection (åœ¨ cross-attention ä¹‹å‰)
        if expert_residual is not None:
            x = x + expert_residual  # [B, T, d_model] + [B, 1, d_model] (broadcast)
            x = self.norm_expert(x)

        # Pre-LN Cross-Attention with Audio Gate
        gate = self.audio_gate(memory.mean(dim=1, keepdim=True))
        residual = x
        x = self.norm2(x)
        x = self.cross_attn(x, memory, memory, key_padding_mask=memory_key_padding_mask, need_weights=False)[0]
        x = residual + gate * x

        # Pre-LN FFN
        residual = x
        x = self.norm3(x)
        x = self.ffn(x)
        return residual + x


# ============================================================
# Differentiable ReAct Decoder â€” ä¸»å…¥å£
# ============================================================

class DiffReActDecoder(nn.Module):
    """
    å¯å¾®ç«¯åˆ°ç«¯ ReAct Decoderã€‚

    ç›¸æ¯” v3.2 çš„ ReActDecoder (ç¦»æ•£å·¥å…·è°ƒç”¨):
      âœ… ç«¯åˆ°ç«¯å¯è®­ç»ƒ (differentiable)
      âœ… è¿ç»­ä¸“å®¶èåˆ (continuous feature fusion)
      âœ… å®Œæ•´æ¢¯åº¦å›ä¼  (full gradient flow)
      âœ… æ”¯æŒè®­ç»ƒå’Œæ¨ç†åŒæ¨¡å¼

    æ¨ç†æ—¶ä»ç„¶å¯ä»¥è¾“å‡ºäººç±»å¯è¯»çš„"æ¨ç†æ­¥éª¤"ï¼Œ
    è®­ç»ƒæ—¶ä½¿ç”¨ teacher forcing + è¾…åŠ©æŸå¤±ã€‚
    """

    def __init__(
        self,
        language_decoder: 'LanguageDecoder',
        emotion_head: nn.Module,
        genre_head: nn.Module,
        instrument_head: nn.Module,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        vocab_size: int = 50257,
        max_seq_len: int = 128,
        tokenizer: Any = None,
    ):
        super().__init__()

        # ä¿ç•™è¯­è¨€è§£ç å™¨çš„ embedding å’Œ lm_head
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # weight tying

        # ä½ç½®ç¼–ç 
        self.pos_encoder = self._build_pos_encoding(d_model, max_seq_len)

        # Decoder layers (expert-aware)
        self.layers = nn.ModuleList([
            DiffDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # å› æœæ©ç 
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.full((max_seq_len, max_seq_len), float("-inf")), diagonal=1),
            persistent=False,
        )
        self.max_seq_len = max_seq_len

        # --- å¯å¾®ä¸“å®¶å·¥å…· ---
        self.emotion_tool = DiffEmotionTool(emotion_head)
        self.genre_tool = DiffGenreTool(genre_head)
        self.instrument_tool = DiffInstrumentTool(instrument_head)
        self.tempo_tool = DiffTempoTool()

        # --- Soft Router ---
        self.router = SoftRouter(d_model, num_experts=4)

        # --- Expert Feature Fuser ---
        self.fuser = ExpertFeatureFuser(d_model)

        # --- é‡æ„åçš„ decoderï¼Œåˆå§‹åŒ–å‚æ•° ---
        # ä»é¢„è®­ç»ƒçš„ language_decoder å¤åˆ¶å‚æ•°
        self._init_from_pretrained(language_decoder)

        self.tokenizer = tokenizer
        self._init_weights()

    def _build_pos_encoding(self, d_model: int, max_len: int):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        pos_encoder = nn.Module()
        pos_encoder.register_buffer("pe", pe)
        pos_encoder.forward = lambda x: x + pos_encoder.pe[:, :x.size(1), :]
        return pos_encoder

    def _init_from_pretrained(self, pretrained_decoder):
        """ä»é¢„è®­ç»ƒçš„ LanguageDecoder å¤åˆ¶æƒé‡"""
        if pretrained_decoder is None:
            return

        # å¤åˆ¶ token embedding
        try:
            self.token_embedding.weight.data.copy_(pretrained_decoder.token_embedding.weight.data)
        except Exception:
            pass

        # å¤åˆ¶ lm_head (weight tying handles it)
        try:
            self.lm_head.weight.data.copy_(pretrained_decoder.lm_head.weight.data)
        except Exception:
            pass

        # å¤åˆ¶ decoder layer æƒé‡ï¼ˆæ³¨æ„ DiffDecoderLayer å¤šäº† expert ç›¸å…³éƒ¨åˆ†ï¼‰
        try:
            for i, layer in enumerate(pretrained_decoder.layers):
                if i >= len(self.layers):
                    break
                own_layer = self.layers[i]
                own_layer.self_attn.load_state_dict(layer.self_attn.state_dict())
                own_layer.cross_attn.load_state_dict(layer.cross_attn.state_dict())
                own_layer.ffn.load_state_dict(layer.ffn.state_dict())
                own_layer.norm1.load_state_dict(layer.norm1.state_dict())
                own_layer.norm2.load_state_dict(layer.norm2.state_dict())
                own_layer.norm3.load_state_dict(layer.norm3.state_dict())
                own_layer.audio_gate.load_state_dict(layer.audio_gate.state_dict())
        except Exception as e:
            print(f"  [DiffReActDecoder] Partial pretrained weights loaded: {e}")

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and "token_embedding" not in name and "lm_head" not in name:
                if "emotion_tool" not in name and "genre_tool" not in name and "instrument_tool" not in name:
                    nn.init.xavier_uniform_(p, gain=0.5)

    def compute_expert_features(
        self,
        memory: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        è®¡ç®—æ‰€æœ‰ 4 ä¸ªä¸“å®¶çš„ç‰¹å¾ï¼ˆä¿æŒè®¡ç®—å›¾ï¼‰ã€‚

        Returns:
            (emotion_feat, genre_feat, instrument_feat, tempo_feat)
            å„ç»´åº¦: [B, 4], [B, 10], [B, 10], [B, 3]
        """
        emo_feat = self.emotion_tool(memory)
        gen_feat = self.genre_tool(memory)
        ins_feat = self.instrument_tool(memory)
        tmp_feat = self.tempo_tool(memory)
        return emo_feat, gen_feat, ins_feat, tmp_feat

    def forward(
        self,
        memory: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
        return_expert_losses: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        å¯å¾®å‰å‘ä¼ æ’­ (è®­ç»ƒ & æ¨ç†)ã€‚

        Args:
            memory: backbone è¾“å‡º [B, T_audio, d_model]
            input_ids: [B, T_text] (è®­ç»ƒ: teacher forcing)
            attention_mask: [B, T_text]
            return_weights: æ˜¯å¦è¿”å› router æƒé‡
            return_expert_losses: æ˜¯å¦è¿”å›å„ expert çš„è¾…åŠ©æŸå¤±

        Returns:
            outputs: {
                'logits': [B, T, V] â€” decoder è¾“å‡º logits
                'weights': [B, 4] â€” router æƒé‡ (å¯é€‰)
                'expert_losses': dict â€” å„ expert è¾…åŠ©æŸå¤± (å¯é€‰)
                'expert_features': dict â€” å„ expert åŸå§‹ç‰¹å¾ (å¯é€‰)
            }
        """
        B, T = input_ids.shape
        device = input_ids.device

        # --- è®¡ç®—æ‰€æœ‰ expert ç‰¹å¾ï¼ˆåªè®¡ç®—ä¸€æ¬¡ï¼‰ ---
        emo_feat, gen_feat, ins_feat, tmp_feat = self.compute_expert_features(memory)
        # emo_feat: [B, 4], gen_feat: [B, 10], ins_feat: [B, 10], tmp_feat: [B, 3]

        # --- Token Embedding ---
        x = self.token_embedding(input_ids)       # [B, T, d_model]
        x = self.pos_encoder(x)
        x = self.dropout(x)

        # å› æœæ©ç 
        causal_mask = self.causal_mask[:T, :T] if T <= self.max_seq_len else None

        # --- Decoder Layers with Expert Injection ---
        expert_residual = None  # ç¬¬ä¸€å±‚æ²¡æœ‰ expert injection

        for i, layer in enumerate(self.layers):
            # åªåœ¨æœ€åä¸€å±‚è®¡ç®— router æƒé‡å¹¶æ³¨å…¥ expert ç‰¹å¾
            # (è®©å‰é¢çš„å±‚å…ˆå¤„ç†åŸºæœ¬çš„è¯­è¨€ç†è§£)
            if i == len(self.layers) - 1:
                # ä»å½“å‰ hidden state è®¡ç®— router æƒé‡
                router_weights = self.router(x)              # [B, 4]

                # åŠ æƒèåˆä¸“å®¶ç‰¹å¾
                expert_residual = self.fuser(
                    router_weights, emo_feat, gen_feat, ins_feat, tmp_feat, x
                )  # [B, 1, d_model]

            x = layer(
                x, memory,
                expert_residual=expert_residual,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=attention_mask.eq(0) if attention_mask is not None else None,
            )

        x = self.norm(x)                                     # [B, T, d_model]
        logits = self.lm_head(x)                             # [B, T, V]

        outputs = {'logits': logits}

        if return_weights:
            outputs['weights'] = router_weights

        if return_expert_losses:
            outputs['expert_features'] = {
                'emotion': emo_feat,
                'genre': gen_feat,
                'instrument': ins_feat,
                'tempo': tmp_feat,
            }

        return outputs

    # ============================================================
    # æ¨ç†æ¥å£ (è‡ªå›å½’ç”Ÿæˆ)
    # ============================================================

    @torch.no_grad()
    def reason(
        self,
        memory: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 50,
        eos_token_id: int = 50256,
        return_human_readable: bool = True,
    ) -> Union[str, List[int]]:
        """
        æ¨ç†æ—¶çš„è‡ªå›å½’ç”Ÿæˆï¼ˆä»å¯ä½¿ç”¨ @torch.no_gradï¼Œå› ä¸ºæ¨ç†ä¸éœ€è¦æ¢¯åº¦ï¼‰ã€‚

        Args:
            memory: [1, T_audio, d_model]
            max_new_tokens: æœ€å¤§ç”Ÿæˆ token æ•°
            temperature: é‡‡æ ·æ¸©åº¦
            top_k: top-k é‡‡æ ·æ•°é‡
            return_human_readable: Trueâ†’è¿”å›æ–‡æœ¬, Falseâ†’è¿”å› token IDs

        Returns:
            æ–‡æœ¬æè¿°ï¼ˆåŒ…å«æ¨ç†æ­¥éª¤ï¼‰
        """
        self.eval()
        B = memory.size(0)
        device = memory.device

        # --- é¢„è®¡ç®— expert ç‰¹å¾ï¼ˆåªç®—ä¸€æ¬¡ï¼‰ ---
        emo_feat, gen_feat, ins_feat, tmp_feat = self.compute_expert_features(memory)

        # --- åˆå§‹ prompt ---
        prompt = "Analyze this music step by step:\n"
        if self.tokenizer is not None:
            prompt_ids = self.tokenizer.encode(prompt)
        else:
            # Fallback: use a simple BOS
            prompt_ids = [464]  # common GPT-2 BOS

        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)

        generated = []

        for _ in range(max_new_tokens):
            if input_ids.size(1) > self.max_seq_len:
                input_ids = input_ids[:, -self.max_seq_len:]
                attention_mask = attention_mask[:, -self.max_seq_len:]

            # --- Forward (ä½¿ç”¨é¢„è®¡ç®—çš„ expert ç‰¹å¾) ---
            x = self.token_embedding(input_ids)
            x = self.pos_encoder(x)
            x = self.dropout(x)

            T = x.size(1)
            causal_mask = self.causal_mask[:T, :T] if T <= self.max_seq_len else None

            expert_residual = None
            for i, layer in enumerate(self.layers):
                if i == len(self.layers) - 1:
                    router_weights = self.router(x)
                    expert_residual = self.fuser(
                        router_weights, emo_feat, gen_feat, ins_feat, tmp_feat, x
                    )
                x = layer(
                    x, memory,
                    expert_residual=expert_residual,
                    tgt_mask=causal_mask,
                    tgt_key_padding_mask=attention_mask.eq(0) if ßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßß