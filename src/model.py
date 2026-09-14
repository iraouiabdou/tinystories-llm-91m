"""Llama-style decoder-only transformer: RoPE + RMSNorm (pre-norm) + SwiGLU.

Written from scratch on top of `nn.Linear` / `nn.Embedding` / `nn.RMSNorm`.
The only fused kernel used is `F.scaled_dot_product_attention` for efficiency reasons.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Rotary position embeddings

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    cos, sin = cos[None, None], sin[None, None]      # (1, 1, seq_len, head_dim)
    in_dtype = q.dtype
    q, k = q.float(), k.float()                      # rotate in fp32 for stability and more precision
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot.to(in_dtype), k_rot.to(in_dtype)


class RoPE(nn.Module):
    """Precomputed cos/sin tables. Built to `max_seq`, sliced at `offset` so
    that KV-cached decoding gets the right absolute positions."""

    def __init__(self, head_dim: int, max_seq: int = 1024, base: float = 10_000.0):
        super().__init__()
        assert head_dim % 2 == 0, f"head dimension {head_dim} must be even"
        self.max_seq = max_seq
        inv = base ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        freqs = torch.outer(torch.arange(max_seq, dtype=torch.float32), inv)
        emb = torch.cat((freqs, freqs), dim=-1)      # (max_seq, head_dim)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, seq_len: int, offset: int = 0):
        """Slices out the positions we actually need"""
        end = offset + seq_len
        if end > self.max_seq:
            raise ValueError(
                f"position {end} exceeds RoPE table length {self.max_seq}; "
                "raise `rope_max_seq` in the config"
            )
        s = slice(offset, end)
        return self.cos[s], self.sin[s]


# Blocks

class CausalMHA(nn.Module):
    def __init__(self, d_model: int, h: int):
        super().__init__()
        assert d_model % h == 0, f"{d_model} is not divisible by {h}"
        self.d_model, self.h, self.d_k = d_model, h, d_model // h
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, cos, sin, cache=None):
        # x: (B, seq_len, d_model)
        def split_heads(t):
            return t.reshape(t.shape[0], t.shape[1], self.h, self.d_k).swapaxes(1, 2)

        q, k, v = (split_heads(w(x)) for w in (self.w_q, self.w_k, self.w_v))
        q, k = apply_rope(q, k, cos, sin)

        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v

        # During cached decode q has length 1 and attends to everything, so the
        # causal mask is only needed when we feed more than one token at a time.
        out = F.scaled_dot_product_attention(q, k, v, None, 0.0, is_causal=q.shape[2] > 1)
        out = out.swapaxes(1, 2).reshape(out.shape[0], out.shape[2], self.d_model)
        return self.w_o(out)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        # 8/3 * d_model, rounded to a multiple of 256 for tensor-core friendliness
        self.d_ffn = round((8 * d_model) / (3 * 256)) * 256
        self.w_g = nn.Linear(d_model, self.d_ffn, bias=False)
        self.w_v = nn.Linear(d_model, self.d_ffn, bias=False)
        self.w_o = nn.Linear(self.d_ffn, d_model, bias=False)

    def forward(self, x):
        return self.w_o(F.silu(self.w_g(x)) * self.w_v(x))


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, h: int):
        super().__init__()
        self.causal_mha = CausalMHA(d_model, h)
        self.swiglu = SwiGLU(d_model)
        self.rmsnorm1 = nn.RMSNorm(d_model)
        self.rmsnorm2 = nn.RMSNorm(d_model)

    def forward(self, x, cos, sin, cache=None):
        h = x + self.causal_mha(self.rmsnorm1(x), cos, sin, cache)
        return h + self.swiglu(self.rmsnorm2(h))


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #

class LM(nn.Module):
    def __init__(self, d_model, vocab_size, h, N, max_seq, rope_max):
        super().__init__()
        self.d_model = d_model
        self.max_seq = max_seq
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, h) for _ in range(N)])
        self.rmsnorm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)   # note that bias=False is important for weight tying
        self.lm_head.weight = self.embed.weight          # weight tying
        self.rope = RoPE(d_model // h, rope_max)
        self._init_weights()

    def _init_weights(self):
        """N(0, 1/sqrt(d_model)), with residual-projection weights scaled down by
        1/sqrt(2N) so the residual stream doesn't blow up with depth (GPT-2 trick)."""
        std = self.d_model ** -0.5
        scaled_std = std / (2 * len(self.blocks)) ** 0.5
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and name.endswith("w_o"):
                nn.init.normal_(module.weight, mean=0.0, std=scaled_std)
            elif isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=std)

    @classmethod
    def from_config(cls, cfg: dict, vocab_size: int):
      """Allows to construct the model simply from a dict without unpacking it"""
        return cls(cfg["d_model"], vocab_size, cfg["h"], cfg["N"],
                   cfg["max_seq"], cfg["rope_max_seq"])

    def forward(self, x, past_len: int = 0, caches=None):
        x = self.embed(x)
        cos, sin = self.rope(x.size(1), offset=past_len)
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, None if caches is None else caches[i])
        return self.lm_head(self.rmsnorm(x))
