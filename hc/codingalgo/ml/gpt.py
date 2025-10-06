import torch

import torch.nn.functional as F

from dataclass import dataclass
from torch import nn
from typing import Optional


@dataclass
class GPTConfig:
    embed_dim: int = 256
    window_size: int = 512
    num_head: int = 6
    num_layer: int = 6
    vocab_size: int = 1024


def create_4d_causal_mask(padding_mask: torch.Tensor) -> torch.Tensor:
    # (B, 1, C, C)
    B, C = padding_mask.shape # 0, 1 -> 1 means no padding, 0 means padding
    query_mask = padding_mask.view(B, 1, C, 1).to(torch.bool)
    key_mask = padding_mask.view(B, 1, 1, C).to(torch.bool)
    combined = query_mask & key_mask
    attention_mask = torch.zeros(B, 1, C, C)
    attention_mask = attention_mask.masked_fill(~combined, float("-inf"))
    return attention_mask


class CausalSelfAttention(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.embed_dim % config.num_head == 0
        self.head_dim = config.embed_dim / config.num_head
        self.scaling = self.head_dim ** -0.5
        self.embed_dim = config.embed_dim
        self.qkv_proj = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.register_buffer("causal_mask", 
                             torch.tril(torch.ones(config.window_size, config.window_size))
                             .view(1, 1, config.window_size, config.window_size).to(torch.bool))
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, D = hidden_states.shape
        qkv = self.qkv_proj(hidden_states) # B, C, 3 * D
        query, key, value = qkv.split(self.embed_dim, dim=2)
        query = query.view(B, C, -1, self.head_dim).transpose(1, 2) # B, HN, C, HD
        key = key.view(B, C, -1, self.head_dim).transpose(1, 2) # B, HN, C, HD
        value = value.view(B, C, -1, self.head_dim).transpose(1, 2) # B, HN, C, HD

        attn_weights = torch.matmul(query, key.transpose(-1, -2)) * self.scaling # B, HN, C, C
        if attention_mask is None:
            attention_mask = torch.zeros(B, 1, C, C)
        attention_mask = attention_mask.masked_fill(~self.causal_mask, float("-inf"))
        # attention_mask is a 4D tensor which is (B, 1, C, C), the same attention mask broadcast to all heads
        attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_outputs = torch.matmul(attn_weights, value) # B, HN, C, HD
        attn_outputs = attn_outputs.transpose(1, 2).contiguous().view(B, C, self.embed_dim)
        attn_outputs = self.o_proj(attn_outputs)
        return attn_outputs


class MLP(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.up_proj = nn.Linear(self.embed_dim, 4 * self.embed_dim)
        self.gate = nn.GELU(approximate="tanh")
        self.down_proj = nn.Linear(4 * self.embed_dim, self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up_proj(x)
        x = self.gate(x)
        x = self.down_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.attn = CausalSelfAttention(config)
        self.mlp_layer_norm = nn.LayerNorm(self.embed_dim)
        self.mlp = MLP(config)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.attn_layer_norm(hidden_states), attention_mask)
        hidden_states = hidden_states + self.mlp(self.mlp(hidden_states))
        return hidden_states
    

class GPT(nn.Module):
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.embed_dim),
            wpe = nn.Embedding(config.window_size, config.embed_dim),
            h = nn.ModuleList([Block(config) for _ in range(config.num_layer)]),
            ln = nn.LayerNorm(config.embed_dim),
        ))

        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, C = idx.shape
        position_ids = torch.arrange(0, C, dtype=torch.long)
        tok_emb = self.transformer.wte(idx) # B, C, emb_dim
        pos_emb = self.transformer.wpe(position_ids) # 1, C, emb_dim
        hidden_states = tok_emb + pos_emb
        for block in self.transformer.h:
            hidden_states = block(hidden_states)
        hidden_states = self.transformer.ln(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits