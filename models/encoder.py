from __future__ import annotations

import math

import torch
import torch.nn as nn


def gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionwiseFeedForward(nn.Module):
    """SDT-style position-wise feed-forward with residual connection."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inter = self.dropout_1(gelu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class MultiHeadedAttention(nn.Module):
    """Manual multi-head attention matching the SDT implementation style."""

    def __init__(self, head_count: int, model_dim: int, dropout: float = 0.1):
        super().__init__()
        if model_dim % head_count != 0:
            raise ValueError(f'model_dim={model_dim} must be divisible by head_count={head_count}.')
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim
        self.head_count = head_count
        self.linear_k = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_v = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_q = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(model_dim, model_dim)

    def forward(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        query: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = key.size(0)
        head_count = self.head_count
        dim_per_head = self.dim_per_head

        key = self.linear_k(key).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        query = self.linear_q(query).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))
        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask.bool()
            if mask.ndim == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.ndim == 3:
                mask = mask.unsqueeze(1)
            mask = mask.expand_as(scores)
            scores = scores.masked_fill(mask, -1e10)

        attn = self.softmax(scores)
        attn = self.dropout(attn)
        context = torch.matmul(attn, value)
        avg_attn = attn.mean(dim=1)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, head_count * dim_per_head)
        output = self.linear(context)
        return output, avg_attn


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor, speaker_emb: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        pos_emb = self.pe[:, :length]
        return x + pos_emb + speaker_emb


class TransformerEncoderLayer(nn.Module):
    """SDT-style Transformer encoder layer with pre-norm on deeper layers."""

    def __init__(self, d_model: int, heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadedAttention(heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter_idx: int, inputs: torch.Tensor, pad_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normed = self.layer_norm(inputs) if iter_idx != 0 else inputs
        context, attn = self.self_attn(normed, normed, normed, mask=pad_mask)
        out = self.dropout(context) + inputs
        out = self.feed_forward(out)
        return out, attn


class TransformerEncoder(nn.Module):
    """Unimodal encoder using the SDT transformer design."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        heads: int,
        layers: int,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        self.layers = layers
        self.pos_emb = PositionalEncoding(d_model, max_len=max_len)
        self.transformer_layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout) for _ in range(layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        speaker_emb: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        valid = mask.unsqueeze(-1)
        x = self.pos_emb(x, speaker_emb)
        x = self.dropout(x)
        x = x * valid
        pad_mask = mask.eq(0)
        attn_maps: list[torch.Tensor] = []
        for i, layer in enumerate(self.transformer_layers):
            x, attn = layer(i, x, pad_mask)
            x = x * valid
            attn_maps.append(attn)
        attention = torch.stack(attn_maps, dim=1) if attn_maps else x.new_zeros(x.size(0), 0, x.size(1), x.size(1))
        return {'state': x, 'attention_weights': attention}
