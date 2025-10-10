import torch
import torch.nn.functional as F

from torch import nn
from typing import Optional
from dataclasses import dataclass


def create_causal_mask(B: int, T: int, device: torch.device, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    # B, HN, T, T
    # B, 1, T, T
    causal_mask = torch.tril(torch.ones(T, T, device=device).view(1, 1, T, T).bool())
    if attn_mask is not None:
        query_mask = attn_mask[:, None, :, None].bool()
        key_mask = attn_mask[:, None, None, :].bool()
        causal_mask = causal_mask & query_mask & key_mask
    mask = torch.zeros(B, 1, T, T, device = device).masked_fill(~causal_mask, float("-inf"))
    return mask

@dataclass
class GPTConfig:
    embed_dim: int = 256
    num_head: int = 3
    vocab_size: int = 512
    num_layer: int = 6
    block_size: int = 1000


class Attention(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.num_head = config.num_head
        self.head_dim = self.embed_dim // self.num_head
        assert self.embed_dim == self.head_dim * self.num_head

        self.qkv_proj = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.scaling = self.head_dim ** -0.5

    def forward(self, hidden_states: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        query, key, value = qkv.split(self.embed_dim, dim=-1) # B, T, C
        query = query.view(B, T, self.num_head, self.head_dim).transpose(1, 2) # B, HN, T, HD
        key = key.view(B, T, self.num_head, self.head_dim).transpose(1, 2)
        value = value.view(B, T, self.num_head, self.head_dim).transpose(1, 2)

        attn_weights = query @ key.transpose(-1, -2) * self.scaling
        attn_mask = create_causal_mask(B, T, hidden_states.device, attn_mask)
        attn_weights = attn_weights + attn_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_outputs = attn_weights @ value
        attn_outputs = attn_outputs.transpose(1, 2).contiguous().view(B, T, C)

        attn_outputs = self.o_proj(attn_outputs)
        return attn_outputs

class MLP(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.up_proj = nn.Linear(self.embed_dim, 4 * self.embed_dim)
        self.gate = nn.GELU()
        self.down_proj = nn.Linear(4 * self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.gate(self.up_proj(hidden_states)))

class Decoder(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attn = Attention(config)
        self.attn_layer_norm = nn.LayerNorm(config.embed_dim)
        self.mlp = MLP(config)
        self.mlp_layer_norm = nn.LayerNorm(config.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.attn_layer_norm(hidden_states))
        hidden_states = hidden_states + self.mlp(self.mlp_layer_norm(hidden_states))
        return hidden_states
    
class GPT(nn.Module):
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.embed_dim),
            wpe = nn.Embedding(config.block_size, config.embed_dim),
            h = nn.ModuleList([Decoder(config) for _ in range(config.num_layer)]),
            ln_f = nn.LayerNorm(config.embed_dim),
        ))

        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size)

    
    def forward(self, idx: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = idx.shape
        wte_emb = self.transformer.wte(idx)
        position_ids = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, T)
        pos_emb = self.transformer.wpe(position_ids)

        hidden_states = wte_emb + pos_emb
        for block in self.transformer.h:
            hidden_states = block(hidden_states)

        hidden_states = self.transformer.ln_f(hidden_states)
        
        logits = self.lm_head(hidden_states)
        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), target.view(-1))
        return logits, loss