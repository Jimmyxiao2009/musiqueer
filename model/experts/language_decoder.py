"""
MusicUnderstand 3.0 — Language Decoder Head（v3.1: Audio Gate）
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class DecoderLayer(nn.Module):
    """Pre-LN Decoder Layer with Audio Gate"""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.audio_gate = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model), nn.Dropout(dropout),
        )

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Pre-LN Self-Attention
        residual = tgt
        x = self.norm1(tgt)
        x = self.self_attn(x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask, need_weights=False)[0]
        x = residual + x

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


class LanguageDecoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024, vocab_size=50257, max_seq_len=128, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_seq_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("causal_mask", torch.triu(torch.full((max_seq_len, max_seq_len), float("-inf")), diagonal=1), persistent=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        for name, p in self.named_parameters():
            if p.dim() > 1 and "token_embedding" not in name and "lm_head" not in name:
                nn.init.xavier_uniform_(p, gain=0.5)

    def forward(self, memory, input_ids, attention_mask=None, memory_mask=None):
        B, T = input_ids.shape
        x = self.token_embedding(input_ids)
        x = self.pos_encoder(x)
        x = self.dropout(x)

        causal_mask = self.causal_mask[:T, :T] if T <= self.max_seq_len else torch.triu(
            torch.full((T, T), float("-inf"), device=input_ids.device), diagonal=1)

        for layer in self.layers:
            x = layer(x, memory, tgt_mask=causal_mask, tgt_key_padding_mask=attention_mask.eq(0) if attention_mask is not None else None)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(self, memory, tokenizer, max_new_tokens=128, temperature=0.8, top_k=50, eos_token_id=50256, bos_token_id=50256):
        self.eval()
        B = memory.size(0)
        device = memory.device
        input_ids = torch.full((B, 1), 464, dtype=torch.long, device=device)
        attention_mask = torch.ones((B, 1), dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            if input_ids.size(1) > self.max_seq_len:
                input_ids = input_ids[:, -self.max_seq_len:]
                attention_mask = attention_mask[:, -self.max_seq_len:]
            logits = self.forward(memory, input_ids, attention_mask)
            next_logits = logits[:, -1, :] / temperature

            if top_k > 0:
                topk_v, topk_i = torch.topk(next_logits, top_k, dim=-1)
                next_logits = torch.full_like(next_logits, float("-inf"))
                next_logits.scatter_(-1, topk_i, topk_v)

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones((B, 1), device=device, dtype=torch.long)], dim=-1)
            finished = finished | (next_token.squeeze(-1) == eos_token_id)
            if finished.all():
                break

        generated = []
        for i in range(B):
            ids = input_ids[i].tolist()
            if eos_token_id in ids:
                ids = ids[:ids.index(eos_token_id)]
            text = tokenizer.decode(ids).strip()
            generated.append(text)
        return generated
