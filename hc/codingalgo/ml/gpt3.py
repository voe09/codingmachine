
import tiktoken
import torch
import torch.nn.functional as F

from torch import nn
from typing import Optional
from dataclasses import dataclass


@dataclass
class GPTConfig:
    embed_dim: int =  256
    vocab_size: int = 50257
    num_head: int = 4
    block_size: int = 1000
    num_layer: int = 6


class Attention(nn.Module):
     
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.head_dim = config.embed_dim // config.num_head
        self.num_head = config.num_head
        assert self.num_head * self.head_dim == self.embed_dim

        self.qkv_proj = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.scaling = self.head_dim ** -0.5

        self.register_buffer("tril", torch.tril(torch.ones(1, 1, config.block_size, config.block_size)).bool())

    def forward(self, hidden_states: torch.Tensor):
        B, T, C = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        query, key, value = qkv.split(self.embed_dim, dim=-1) # B, T, C
        query = query.view(B, T, self.num_head, self.head_dim).transpose(1, 2) # B, HN, T, HD
        key = key.view(B, T, self.num_head, self.head_dim).transpose(1, 2)
        value = value.view(B, T, self.num_head, self.head_dim).transpose(1, 2)

        attn_weights = (query @ key.transpose(-1, -2)) * self.scaling # B, HN, T, T
        attn_weights = attn_weights.masked_fill(~self.tril[:, :, :T, :T], float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1) # B, HN, T, T
        
        attn_outputs = attn_weights @ value # B, HN, T, HD
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

    def forward(self, x: torch.Tensor):
        x = self.up_proj(x)
        x = self.gate(x)
        x = self.down_proj(x)
        return x


class Decoder(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.ln1 = nn.LayerNorm(self.embed_dim)
        self.attn = Attention(config)
        self.ln2 = nn.LayerNorm(self.embed_dim)
        self.mlp = MLP(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config
        self.block_size = config.block_size
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.config.vocab_size, self.config.embed_dim),
            wpe = nn.Embedding(self.config.block_size, self.config.embed_dim),
            h = nn.ModuleList([Decoder(self.config) for _ in range(self.config.num_layer)]),
            ln_f = nn.LayerNorm(self.config.embed_dim),
        ))

        self.lm_head = nn.Linear(self.config.embed_dim, self.config.vocab_size)
        self.lm_head.weight = self.transformer.wte.weight
        self.apply(self._init_weight)

    def _init_weight(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, target: Optional[torch.Tensor]) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape

        wte_emb = self.transformer.wte(idx)
        position_ids = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, T)
        wpe_emb = self.transformer.wpe(position_ids)
        
        hidden_states = wte_emb + wpe_emb
        for block in self.transformer.h:
            hidden_states = block(hidden_states)
        hidden_states = self.transformer.ln_f(hidden_states)
        
        logits = self.lm_head(hidden_states) # B, T, V
        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), target.view(-1))
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self, 
        idx: torch.Tensor, 
        max_token: int, 
        temperature: float = 1.0,
        topk: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if generator is None:
            generator = torch.Generator(device=idx.device)
            generator.manual_seed(0)

        for _ in range(max_token):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature
            if topk is not None:
                v, _ = torch.topk(logits, topk)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1, generator=generator)
            idx = torch.cat([idx, next_token], dim=1)
        return idx




         


    