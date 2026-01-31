import torch
import torch.nn.functional as F

from torch import nn
from dataclasses import dataclass
from typing import Optional

@dataclass
class GPTConfig:
    embed_dim: int =  256
    vocab_size: int = 50257
    num_head: int = 4
    block_size: int = 1000
    num_layer: int = 6

class KVCache:

    def __init__(self, block_size: int):
        self.block_size = block_size
        self.k = None
        self.v = None

    def update(self, k: torch.Tensor, v: torch.Tensor):
        if self.k is None:
            self.k = k
            self.v = v
        else:
            self.k = torch.cat([self.k, k], dim=1)
            self.v = torch.cat([self.v, v], dim=1)
        
        return self.k, self.v
    
    def seqlen(self):
        return 0 if self.k is None else self.k.shape[1]

    def truncate(self):
        if self.k is not None and self.k.shape[1] >= self.block_size:
            self.k = self.k[:, -self.block_size + 1:, :]
            self.v = self.v[:, -self.block_size + 1:, :]

class Attention(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.head_dim = config.embed_dim // config.num_head
        self.num_head = config.num_head
        assert self.head_dim * self.num_head == self.embed_dim

        self.qkv_proj = nn.Linear(config.embed_dim, 3 * config.embed_dim)
        self.o_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.scaling = self.head_dim ** -0.5

        self.register_buffer("tril", torch.tril(torch.ones(1, 1, config.block_size, config.block_size)).bool())

    def forward(self, hidden_states: torch.Tensor, kvc: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        query, key, value = qkv.split(self.embed_dim, dim=-1)
        if kvc:
            key, value = kvc.update(key, value)

        KT = key.shape[1]

        query = query.view(B, T, self.num_head, self.head_dim).transpose(1, 2) # B, HN, T, HD
        key = key.view(B, KT, self.num_head, self.head_dim).transpose(1, 2) 
        value = value.view(B, KT, self.num_head, self.head_dim).transpose(1, 2)

        attn_weights = query @ key.transpose(-1, -2) * self.scaling
        if T > 1:
            attn_weights = attn_weights.masked_fill(~self.tril[:, :, :T, :T], float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up_proj(x)
        x = self.gate(x)
        x = self.down_proj(x)
        return x

class Decoder(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.attn = Attention(config)
        self.ln1 = nn.LayerNorm(self.embed_dim)
        self.mlp = MLP(config)
        self.ln2 = nn.LayerNorm(self.embed_dim)
    
    def forward(self, x: torch.Tensor, kvc: Optional[KVCache] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), kvc=kvc)
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.vocab_size = config.vocab_size
        self.block_size = config.block_size
        self.num_layer = config.num_layer

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(self.vocab_size, self.embed_dim),
            wpe=nn.Embedding(self.block_size, self.embed_dim),
            h=nn.ModuleList([Decoder(config) for _ in range(self.num_layer)]),
            ln_f=nn.LayerNorm(self.embed_dim),
        ))

        self.lm_head = nn.Linear(self.embed_dim, self.vocab_size)

        self.lm_head.weight = self.transformer.wte.weight

        self.apply(self._init_weight)

    def _init_weight(self, module: nn.Module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, target: Optional[torch.Tensor] = None, kvcs: Optional[list[KVCache]] = None) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        wte_emb = self.transformer.wte(idx)

        if kvcs:
            for kvc in kvcs:
                kvc.truncate()
        
        offset = kvcs[0].seqlen() if kvcs else 0
        position_ids = torch.arange(offset, offset+T, device=idx.device).expand(B, T)
        wpe_emb = self.transformer.wpe(position_ids)

        hidden_states = wte_emb + wpe_emb
        if not kvcs:
            for block in self.transformer.h:
                hidden_states = block(hidden_states)
        else:
            for kvc, block in zip(kvcs, self.transformer.h):
                hidden_states = block(hidden_states, kvc=kvc)
        hidden_states = self.transformer.ln_f(hidden_states)

        logits = self.lm_head(hidden_states)
        if target is None:
            return logits, None
        
        loss = F.cross_entropy(logits.view(B*T, self.vocab_size), target.view(B*T))
        return logits, loss

    @torch.no_grad()
    def generate(
        self, 
        idx: torch.Tensor, 
        max_token: int, 
        temperature: float = 1.0, 
        topk: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        kvcs: Optional[list[KVCache]] = None,
    ) -> torch.Tensor:
        if generator is None:
            generator = torch.Generator(device=idx.device)
            # for debugging purpose
            generator.manual_seed(0)

        self.eval()
        for _ in range(max_token):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond, kvcs=kvcs)
            logits = logits[:, -1, :] / temperature
            if topk:
                v, _ = torch.topk(logits, topk)
                logits[logits<v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1, generator=generator)
            idx = torch.cat([idx, next_token], dim=1)
        return idx
    
    @torch.no_grad()
    def generate_with_kvcache(
        self, 
        idx: torch.Tensor, 
        max_token: int, 
        temperature: float = 1.0, 
        topk: Optional[int] = None, 
        generator: Optional[torch.Generator] = None):
        if generator is None:
            generator = torch.Generator(device=idx.device)
            generator.manual_seed(0)

        self.eval()
        kvcs = [KVCache(self.block_size) for _ in range(self.num_layer)]
        # prefill
        idx = self.generate(idx, 1, temperature, topk, generator, kvcs)

        for _ in range(max_token - 1):
            idx_cond = idx[:, -1:]
            logits, _ = self(idx_cond, kvcs=kvcs)
            logits = logits[:, -1, :] / temperature
            if topk:
                v, _ = torch.topk(logits, topk)
                logits[logits<v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1, generator=generator)
            idx = torch.cat([idx, next_token], dim=1)
        return idx