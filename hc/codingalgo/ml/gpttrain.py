import torch
import torch.nn.functional as F

from torch import nn
from dataclasses import dataclass


@dataclass
class GPTConfig:
    embed_dim: int = 32
    num_head: int = 3
    block_size: int = 32
    num_layers: int = 6
    vocab_size: int = 100
    

class CausalSelfAttention(nn.Module):

    def __init__(self, embed_dim: int, num_head: int, block_size: int, bias: bool = False, dropout: float = 0.06):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_head = num_head
        self.head_dim = self.embed_dim // self.num_head
        assert self.embed_dim == self.num_head * self.head_dim

        self.qkv_proj = nn.Linear(embed_dim, 3  * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self.scaling = self.head_dim ** -0.5
        self.dropout = dropout
        
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
            .view(1, 1, block_size, block_size)
            .bool()
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, C = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        query, key, value = qkv.split(self.embed_dim, dim=-1) # B, T, C
        query = query.view(B, T, -1, self.head_dim).transpose(1, 2) # B, HN, T, HD
        key = key.view(B, T, -1, self.head_dim).transpose(1, 2)
        value = value.view(B, T, -1, self.head_dim).transpose(1, 2) # B, HN, T, HD

        attn_weights = query @ key.transpose(-1, -2) * self.scaling
        attn_weights = attn_weights.masked_fill(~self.tril[:, :, :T, :T], float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1) # B, HN, T, T
        attn_weights = F.dropout(attn_weights, self.dropout, training=self.training)
        attn_outputs = attn_weights @ value # B, HN, T, HD
        attn_outputs = attn_outputs.transpose(1, 2).contiguous().view(B, T, C) # B, T, C
        attn_outputs = self.o_proj(attn_outputs) # B, T, C
        return attn_outputs

class MLP(nn.Module):

    def __init__(self, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Decoder(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attn_layer_norm = nn.LayerNorm(config.embed_dim)
        self.attn = CausalSelfAttention(config.embed_dim, config.num_head, config.block_size)
        self.mlp_layer_norm = nn.LayerNorm(config.embed_dim)
        self.mlp = MLP(config.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_layer_norm(x))
        x = x + self.mlp(self.mlp_layer_norm(x))
        return x
    
class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_emb = nn.Embedding(config.block_size, config.embed_dim)

        self.decoders = nn.ModuleList([Decoder(config) for _ in range(config.num_layers)])
        self.final_layer_norm = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T = tokens.shape
        token_emb = self.token_emb(tokens)
        pos = torch.arange(T)
        pos_emb = self.pos_emb(pos)[None, :, :]

        hidden_states = token_emb + pos_emb
        for decoder in self.decoders:
            hidden_states = decoder(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits
    
    def generate(self, idx: torch.Tensor, max_tokens: int) -> torch.Tensor:
        T = self.config.block_size
        for _ in range(max_tokens):
            logits = self(idx[:, -T:])[:, -1, :]
            prob = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(prob, num_samples=1)
            idx = torch.cat([idx, next_token], dim=-1)
        
        return idx
