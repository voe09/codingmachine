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
    num_head: int = 4
    vocab_size: int = 50257
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

        # weight tie
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    
    def _init_weights(self, module):
      if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
          torch.nn.init.zeros_(module.bias)
      elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    
    def forward(self, idx: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = idx.shape
        wte_emb = self.transformer.wte(idx)
        position_ids = torch.arange(0, T, device=idx.device).unsqueeze(0).expand(B, T)
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
    

with open('input.txt', 'r') as f:
  text = f.read()

import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode(text)

class DataLoader:
  def __init__(self, B, T):
    self.B = B
    self.T = T
    
    with open('input.txt', 'r') as f:
      text = f.read()
    
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(text)
    self.tokens = torch.tensor(tokens)
    self.current_pos = 0

  def next_batch(self):
    B, T = self.B, self.T
    buf = self.tokens[self.current_pos : self.current_pos + B * T + 1]
    x = buf[:-1].view(B, T)
    y = buf[1:].view(B, T)
    self.current_pos += B * T
    if self.current_pos + (B + T + 1) > len(self.tokens):
      self.current_pos = 0
    return x, y


model = GPT(GPTConfig())
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
train_loader = DataLoader(B=4, T = 32)
for i in range(50):
  x, y = train_loader.next_batch()
  optimizer.zero_grad()
  logits, loss = model(x, y)
  loss.backward()
  optimizer.step()
  print(f"step {i}, loss: {loss.item()}")