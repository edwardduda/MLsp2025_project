import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.kan_layer import KANLayer
from kan_lm.config import KANLMConfig


class CausalAttention(nn.Module):
    """Standard multi-head causal self-attention."""

    def __init__(self, config: KANLMConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model

        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.context_length, config.context_length))
            .view(1, 1, config.context_length, config.context_length),
        )

    def forward(self, x):
        B, T, D = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.d_model, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.resid_dropout(self.out_proj(out))


class KANFeedForward(nn.Module):
    """
    Feed-forward block using KAN layers instead of MLP.

    Replaces the standard  Linear -> GELU -> Linear  with two KAN layers
    whose per-edge B-spline functions serve as both the linear map and
    the learned nonlinearity.
    """

    def __init__(self, config: KANLMConfig):
        super().__init__()
        self.kan1 = KANLayer(
            in_features=config.d_model,
            out_features=config.kan_hidden,
            num_control_points=config.num_control_points,
            degree=config.spline_degree,
            range_min=config.spline_range_min,
            range_max=config.spline_range_max,
        )
        self.kan2 = KANLayer(
            in_features=config.kan_hidden,
            out_features=config.d_model,
            num_control_points=config.num_control_points,
            degree=config.spline_degree,
            range_min=config.spline_range_min,
            range_max=config.spline_range_max,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, D = x.shape
        input_dtype = x.dtype
        x = x.reshape(B * T, D)
        x = self.kan1(x.float())
        x = self.kan2(x).to(dtype=input_dtype)
        x = x.reshape(B, T, -1)
        return self.dropout(x)


class KANTransformerBlock(nn.Module):
    """Pre-norm transformer block: LN -> Attn -> Residual, LN -> KAN FFN -> Residual."""

    def __init__(self, config: KANLMConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalAttention(config=config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = KANFeedForward(config=config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class KANLanguageModel(nn.Module):
    """
    GPT-style causal language model with KAN (B-Spline) feed-forward layers.

    Architecture:
        Token Embedding + Positional Embedding
        -> N x KANTransformerBlock
        -> LayerNorm
        -> Linear head (tied with token embeddings)
    """

    def __init__(self, config: KANLMConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.context_length, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            KANTransformerBlock(config=config) for _ in range(config.n_layers)
        ])

        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share token embedding weights with output head
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids, targets=None):
        """
        Args:
            input_ids: (batch, seq_len) token indices
            targets:   (batch, seq_len) target token indices for loss computation

        Returns:
            logits: (batch, seq_len, vocab_size)
            loss:   scalar cross-entropy loss (only if targets provided)
        """
        B, T = input_ids.shape
        assert T <= self.config.context_length

        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.emb_dropout(self.token_emb(input_ids) + self.pos_emb(positions))

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
