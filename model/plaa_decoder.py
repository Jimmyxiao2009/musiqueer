"""
MusicUnderstand 3.0 — PLAA 500M Decoder
Architecture: 24-layer GQA (16h/8kv) Decoder with Emotion Cross-Attention + SwiGLU + RoPE + RMSNorm
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ── Normalization ──────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


# ── Rotary Position Embedding ─────────────────────────────────────────────────

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class RotaryPositionalEmbedding(nn.Module):
    """RoPE — pre-computed cos/sin cache"""
    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        device = self.inv_freq.device
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, head_dim]
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def get_cos_sin(self, seq_len: int, device: torch.device):
        """Return cos, sin of shape [1, 1, seq_len, head_dim]"""
        if seq_len > self.cos_cached.shape[2]:
            self._build_cache(seq_len)
        return (self.cos_cached[:, :, :seq_len, :].to(device, non_blocking=True),
                self.sin_cached[:, :, :seq_len, :].to(device, non_blocking=True))

    def apply(self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """Apply RoPE to Q and K tensors of shape [B, n_head, T, head_dim]"""
        cos = cos.to(q.dtype)
        sin = sin.to(q.dtype)
        q_embed = q * cos + rotate_half(q) * sin
        k_embed = k * cos + rotate_half(k) * sin
        return q_embed, k_embed


# ── Grouped Query Attention (GQA) ─────────────────────────────────────────────

class GroupedQueryAttention(nn.Module):
    """GQA self-attention with RoPE, no bias on projections"""
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE
        if cos is not None and sin is not None:
            # cos/sin: [1, 1, T, head_dim], q/k: [B, n_head, T, head_dim]
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin

        # GQA: repeat KV heads
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            attn = attn + attn_mask
        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


# ── Emotion Cross-Attention ────────────────────────────────────────────────────

class EmotionCrossAttention(nn.Module):
    """
    Cross-attention over emotion memory (e_mem).
    Q = decoder hidden state, K/V = e_mem projection.
    Output gated by sigmoid(Linear(mean(e_mem, dim=1))).
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(d_model, 1, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        e_mem: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        _, T_e, _ = e_mem.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(e_mem).view(B, T_e, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(e_mem).view(B, T_e, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if memory_mask is not None:
            attn = attn.masked_fill(memory_mask[:, None, None, :], float("-inf"))

        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, -1)
        out = self.o_proj(out)

        # Gate: g = sigmoid(Linear(mean(e_mem, dim=1)))  → [B, 1, 1]
        gate = torch.sigmoid(self.gate(e_mem.mean(dim=1, keepdim=True)))
        return gate * out


# ── Decoder Layer ──────────────────────────────────────────────────────────────

class PLAA500MDecoderLayer(nn.Module):
    """
    Pre-LN Decoder Layer for PLAA 500M.
    Layout: Self-Attn (GQA+RoPE) → Emotion Cross-Attn (optional) → SwiGLU FFN
    """
    def __init__(
        self,
        d_model: int = 1024,
        n_heads: int = 16,
        n_kv_heads: int = 8,
        dim_feedforward: int = 2816,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads, dropout)
        self.emotion_cross_attn = EmotionCrossAttention(d_model, n_heads // 2, dropout)

        self.norm1 = RMSNorm(d_model)
        self                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logits = next_logits.masked_fill(indices_to_remove, float("-inf"))

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones((B, 1), dtype=torch.long, device=device)], dim=-1)

            finished = finished | (next_token.squeeze(-1) == eos_token_id)
            if finished.all():
                break

        return input_ids

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Return number of parameters (excluding lm_head/embedding if non_embedding)."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.token_embedding.weight.numel()
        return n


if __name__ == "__main__":
    # Quick sanity test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PLAA500MDecoder(dropout=0.0)
    model.to(device)
    print(f"PLAA 500M Decoder — total params: {model.get_num_params():,}")
    print(f"                  non-embedding: {model.get_num_params(non_embedding=True):,}")

    B, T = 2, 64
    input_ids = torch.randint(0, 64000, (B, T), device=device)
    attn_mask = torch.ones((B, T), dtype=torch.long, device=device)
    e_mem = torch.randn(B, 16, 1024, device=device)

    logits = model(input_ids, attn_mask, e_mem=e_mem, use_emotion=True)
    print(f"Forward output shape: {logits.shape}  (expected [B, T, 64000])")

    # Test generation
    prompt = torch.randint(0, 64000, (B, 8), device=device)
    out = model.generate(prompt, e_mem=e_mem, max_new_tokens=16, temperature=0.8, top_k=50)
    print(f"Generation output shape: {out.shape}")
    print("Sanity check passed OK")
