import torch
import torch.nn.functional as F

from typing import Optional
from torch import nn
from dataclasses import dataclass

@dataclass
class GPTConfig:
    embed_dim: int = 256
    num_head: int = 4
    num_layer: int = 6
    vocab_size: int = 50527
    block_size: int = 1000


class Attention(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config
        self.embed_dim = config.embed_dim
        self.num_head = config.num_head
        self.head_dim = self.embed_dim // self.num_head
        assert self.embed_dim == self.num_head * self.head_dim
        self.block_size = config.block_size

        self.scaling = self.head_dim ** -0.5
        
        self.qkv_proj = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.register_buffer("tril", torch.tril(torch.ones(1, 1, self.block_size, self.block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv_proj(x)
        query, key, value = qkv.split(self.embed_dim, dim=-1) # B, T, C
        query = query.view(B, T, self.num_head, self.head_dim).transpose(1, 2) # B, HN, T, HD
        key = key.view(B, T, self.num_head, self.head_dim).transpose(1, 2) # B, HN, T, HD
        value = value.view(B, T, self.num_head, self.head_dim).transpose(1, 2)

        attn_weights = (query @ key.transpose(-1, -2)) * self.scaling
        attn_weights = attn_weights.masked_fill(self.tril[:, :, :T, :T] == 0, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_outputs = attn_weights @ value # B, HN, T, HD
        attn_outputs = attn_outputs.transpose(1, 2).contiguous().view(B, T, C)
        attn_outputs = self.o_proj(attn_outputs)
        return attn_outputs

class MLP(nn.Module):
    
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config
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

        self.config = config
        self.embed_dim = config.embed_dim
        self.ln1 = nn.LayerNorm(self.embed_dim)
        self.attn = Attention(self.config)
        self.ln2 = nn.LayerNorm(self.embed_dim)
        self.mlp = MLP(self.config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_dim = config.embed_dim
        self.block_size = config.block_size
        self.num_layer = config.num_layer

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.vocab_size, self.embed_dim),
            wpe = nn.Embedding(self.block_size, self.embed_dim),
            h = nn.ModuleList([Decoder(config) for _ in range(self.num_layer)]),
            ln_f = nn.LayerNorm(self.embed_dim),
        ))

        self.lm_head = nn.Linear(self.embed_dim, self.vocab_size)

        self.lm_head.weight = self.transformer.wte.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape

        wte_emb = self.transformer.wte(idx)

        position_ids = torch.arange(0, T, dtype=torch.long, device=idx.device)
        wpe_emb = self.transformer.wpe(position_ids)

        hidden_states = wte_emb + wpe_emb
        for block in self.transformer.h:
            hidden_states = block(hidden_states)

        hidden_states = self.transformer.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, self.vocab_size), targets.view(B * T))
        return logits, loss

    def generate(self, idx: torch.Tensor, max_token: int) -> torch.Tensor:
        for _ in range(max_token):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx

data = torch.randint(0, 100, (1, 32))
X = data[:, :-1]
y = data[:, 1:]

config = GPTConfig(embed_dim=256, num_head=4, num_layer=2, vocab_size=101, block_size=100)
model = GPT(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
for step in range(50):
    model.train()
    optimizer.zero_grad()
    logits, loss = model(X, y)
    loss.backward()
    optimizer.step()
    print(f"step: {step}, loss: {loss.item()}")


import torch
import torch.nn.functional as F

from typing import Optional
from torch import nn
from dataclasses import dataclass

class KVCache:

    def __init__(self, block_size: int):
        self.block_size = block_size
        self.key = None
        self.value = None

    def update(self, key: torch.Tensor, value: torch.Tensor):
        if self.key is None:
            self.key = key
            self.value = value
        else:
            self.key = torch.cat([self.key, key], dim=1)
            self.value = torch.cat([self.value, value], dim=1)
        if self.key.shape[1] > self.block_size:
            self.key = self.key[:, -self.block_size:, :]
            self.value = self.value[:, -self.block_size:, :]
    
    def get(self):
        return self.key, self.value
    
    def seq_len(self):
        return 0 if self.key is None else self.key.shape[1]

@dataclass
class GPTConfig:
    embed_dim: int = 256
    num_head: int = 4
    num_layer: int = 6
    vocab_size: int = 50527
    block_size: int = 1000


class Attention(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config
        self.embed_dim = config.embed_dim
        self.num_head = config.num_head
        self.head_dim = self.embed_dim // self.num_head
        assert self.embed_dim == self.num_head * self.head_dim
        self.block_size = config.block_size

        self.scaling = self.head_dim ** -0.5
        
        self.qkv_proj = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.register_buffer("tril", torch.tril(torch.ones(1, 1, self.block_size, self.block_size)))

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv_proj(x)
        query, key, value = qkv.split(self.embed_dim, dim=-1) # B, T, C
        if kv_cache is not None:
            kv_cache.update(key, value)
            key, value = kv_cache.get()

        key_T = key.shape[1]

        query = query.view(B, T, self.num_head, self.head_dim).transpose(1, 2) # B, HN, T, HD
        key = key.view(B, key_T, self.num_head, self.head_dim).transpose(1, 2) # B, HN, key_T, HD
        value = value.view(B, key_T, self.num_head, self.head_dim).transpose(1, 2)

        attn_weights = (query @ key.transpose(-1, -2)) * self.scaling
        if T > 1:
            attn_weights = attn_weights.masked_fill(self.tril[:, :, :T, :key_T] == 0, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_outputs = attn_weights @ value # B, HN, T, HD
        attn_outputs = attn_outputs.transpose(1, 2).contiguous().view(B, T, C)
        attn_outputs = self.o_proj(attn_outputs)
        return attn_outputs

class MLP(nn.Module):
    
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config
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

        self.config = config
        self.embed_dim = config.embed_dim
        self.ln1 = nn.LayerNorm(self.embed_dim)
        self.attn = Attention(self.config)
        self.ln2 = nn.LayerNorm(self.embed_dim)
        self.mlp = MLP(self.config)

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), kv_cache)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_dim = config.embed_dim
        self.block_size = config.block_size
        self.num_layer = config.num_layer

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.vocab_size, self.embed_dim),
            wpe = nn.Embedding(self.block_size, self.embed_dim),
            h = nn.ModuleList([Decoder(config) for _ in range(self.num_layer)]),
            ln_f = nn.LayerNorm(self.embed_dim),
        ))

        self.lm_head = nn.Linear(self.embed_dim, self.vocab_size)

        self.lm_head.weight = self.transformer.wte.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, kv_caches: Optional[list[KVCache]] = None) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape

        wte_emb = self.transformer.wte(idx)

        position_offset = 0 if kv_caches is None else kv_caches[0].seq_len()
        if position_offset >= self.block_size:
            position_offset = self.block_size - 1
        position_ids = torch.arange(position_offset, position_offset + T, dtype=torch.long, device=idx.device)
        wpe_emb = self.transformer.wpe(position_ids)

        hidden_states = wte_emb + wpe_emb

        if kv_caches is None:
            for block in self.transformer.h:
                hidden_states = block(hidden_states)
        else:
            for block, kvc in zip(self.transformer.h, kv_caches):
                hidden_states = block(hidden_states, kvc)

        hidden_states = self.transformer.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, self.vocab_size), targets.view(B * T))
        return logits, loss

    def generate(self, idx: torch.Tensor, max_token: int, kv_caches: Optional[list[KVCache]] = None) -> torch.Tensor:
        for _ in range(max_token):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond, kv_caches=kv_caches)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx

    def generate_with_kvcache(self, idx: torch.Tensor, max_token: int) -> torch.Tensor:

        kv_caches = [KVCache(self.block_size) for _ in range(self.num_layer)]

        idx = self.generate(idx, 1, kv_caches)

        for _ in range(1, max_token):
            idx_cond = idx[:, -1:]
            logits, _ = self(idx_cond, kv_caches=kv_caches)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx

data = torch.randint(0, 100, (1, 32))
X = data[:, :-1]
y = data[:, 1:]

config = GPTConfig(embed_dim=256, num_head=4, num_layer=2, vocab_size=101, block_size=100)
model = GPT(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
for step in range(50):
    model.train()
    optimizer.zero_grad()
    logits, loss = model(X, y)
    loss.backward()
    optimizer.step()
    print(f"step: {step}, loss: {loss.item()}")


with torch.no_grad():
    model.eval()
    idx = data[:, :-4]
    output = model.generate(idx, 4)
    output_with_kvc = model.generate_with_kvcache(idx, 4)
    print(output)
    print(output_with_kvc)

