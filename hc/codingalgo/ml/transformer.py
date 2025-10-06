import torch
import torch.nn.functional as F

from typing import Optional
from torch import nn
# Non-Multihead Encoder and Decoder 

class Attention(nn.Module):

  def __init__(
      self, 
      embed_dim: int,
      num_head: int, 
      bias: bool = True, 
      dropout: float = 0.06):
    super().__init__()
    self.embed_dim = embed_dim
    self.num_head = num_head
    self.head_dim = embed_dim // num_head
    assert self.head_dim * self.num_head == self.embed_dim

    self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
    self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
    self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
    self.o_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    self.scaling = embed_dim ** -0.5
    self.dropout = dropout

  def forward(
      self,
      hidden_states: torch.Tensor, # (B, tgt_len, embed_dim)
      key_value_states: Optional[torch.Tensor] = None, # (B, src_len, embed_dim)
      attention_mask: Optional[torch.Tensor] = None, # (B, src_len)
    ):
    is_cross_attention = True if key_value_states is not None else False
    bz, tgt_len = hidden_states.shape[:-1]

    query = self.q_proj(hidden_states) # B, tgt_len, D

    query = query.view(bz, tgt_len, -1, self.head_dim).transpose(1, 2)

    current_states = hidden_states if not is_cross_attention else key_value_states
    key = self.k_proj(current_states) # B, src_len, D
    value = self.v_proj(current_states) # B, src_len, D

    src_len = key.shape[1]

    key = key.view(bz, src_len, -1, self.head_dim).transpose(1, 2) # B, num_head, src_len, head_dim
    value = value.view(bz, src_len, -1, self.head_dim).transpose(1, 2)

    attention_weights = torch.matmul(query, key.transpose(-1, -2)) * self.scaling # B, num_head, tgt_len, src_len
    if attention_mask is not None:
      attention_weights = attention_weights + attention_mask

    attention_weights = F.softmax(attention_weights, dim=-1)
    attention_weights = F.dropout(attention_weights, self.dropout, training=self.training)

    attention_outputs = torch.matmul(attention_weights, value) # B, num_head, tgt_len, D
    attention_outputs = self.o_proj(attention_outputs.transpose(1, 2).contiguous().view(bz, tgt_len, self.embed_dim))

    return attention_outputs, attention_weights

class EncoderLayer(nn.Module):

  def __init__(
      self, 
      embed_dim: int, 
      num_head: int, 
      fnn_dim: Optional[int] = None, 
      bias: bool = True, 
      dropout: float = 0.06):
    super().__init__()
    self.embed_dim = embed_dim
    self.fnn_dim = fnn_dim
    self.num_head = num_head
    if self.fnn_dim is None:
      self.fnn_dim = self.embed_dim // 4
    self.self_attn = Attention(embed_dim, num_head, bias, dropout)
    self.self_attn_layer_norm = nn.LayerNorm(embed_dim)

    self.dropout = dropout
    self.activation_fn = nn.SiLU()

    self.fc1 = nn.Linear(embed_dim, self.fnn_dim)
    self.fc2 = nn.Linear(self.fnn_dim, embed_dim)
    self.final_layer_norm = nn.LayerNorm(embed_dim)

  def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
    residual = hidden_states
    hidden_states, attn_weights = self.self_attn(hidden_states, None, attention_mask)
    hidden_states = F.dropout(hidden_states, self.dropout, training=self.training)
    hidden_states = residual + hidden_states
    hidden_states = self.self_attn_layer_norm(hidden_states)

    residual = hidden_states
    hidden_states = self.activation_fn(self.fc1(hidden_states))
    hidden_states = F.dropout(hidden_states, self.dropout, training=self.training)
    hidden_states = self.fc2(hidden_states)
    hidden_states = F.dropout(hidden_states, self.dropout, training=self.training)
    hidden_states = residual + hidden_states
    hidden_states = self.final_layer_norm(hidden_states)
    return hidden_states, attn_weights

class DecoderLayer(nn.Module):

  def __init__(
      self, 
      embed_dim: int, 
      num_head: int, 
      ffn_dim: Optional[int] = None, 
      bias: bool = True, 
      dropout: float = 0.06):
    super().__init__()
    self.num_head = num_head
    self.self_attn = Attention(embed_dim, num_head, bias, dropout)
    self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
    self.dropout = dropout
    self.encoder_attn = Attention(embed_dim, bias, dropout)
    self.encoder_attn_layer_norm = nn.LayerNorm(embed_dim)
    self.ffn_dim = ffn_dim
    if self.ffn_dim is None:
      self.ffn_dim = embed_dim // 4

    self.mlp = nn.Sequential(
        nn.Linear(embed_dim, self.ffn_dim),
        nn.SiLU(),
        nn.Dropout(self.dropout),
        nn.Linear(self.ffn_dim, embed_dim),
        nn.Dropout(self.dropout),
    )
    self.final_layer_norm = nn.LayerNorm(embed_dim)

  def forward(
      self, 
      hidden_states: torch.Tensor, 
      attention_mask: Optional[torch.Tensor] = None,
      encoder_hidden_states: Optional[torch.Tensor] = None,
      encoder_attention_mask: Optional[torch.Tensor] = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    residual = hidden_states

    hidden_states, attention_weights = self.self_attn(hidden_states, None, attention_mask)
    hidden_states = F.dropout(hidden_states, self.dropout, training=self.training)
    hidden_states = residual + hidden_states
    hidden_states = self.self_attn_layer_norm(hidden_states)

    cross_attn_weights = None
    if encoder_hidden_states is not None:
      residual = hidden_states
      hidden_states, cross_attn_weights = self.encoder_attn(hidden_states, encoder_hidden_states, encoder_attention_mask)
      hidden_states = F.dropout(hidden_states, self.dropout, training=self.training)
      hidden_states = residual + hidden_states
      hidden_states = self.encoder_attn_layer_norm(hidden_states)

    residual = hidden_states
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    hidden_states = self.final_layer_norm(hidden_states)

    return hidden_states, attention_weights, cross_attn_weights


class Model(nn.Module):

  def __init__(
      self, 
      embed_dim: int, 
      num_head: int,
      fcc_dim: Optional[int] = None, 
      bias: bool = True, 
      dropout: float = 0.06,
  ):
    super().__init__()
    self.encoder = EncoderLayer(embed_dim, num_head, fcc_dim, bias, dropout)
    self.decoder = DecoderLayer(embed_dim, num_head, fcc_dim, bias, dropout)
    self.lm_head = nn.Linear(embed_dim, embed_dim)

  def forward(self, inputs: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    encoder_hidden_states, _ = self.encoder(inputs, attention_mask)
    hidden_states, _, _ = self.decoder(inputs, attention_mask, encoder_hidden_states, attention_mask)
    outputs = self.lm_head(hidden_states.sum(dim=1))
    return outputs

n = 10000
embed_dim = 32
seq_len = 10

X = torch.randn((n, seq_len, embed_dim))

# initial random mask
mask = (torch.randn((n, seq_len)) > 0.1).int()

# ensure each row has at least one "1"
row_has_no_ones = mask.sum(dim=1) == 0
if row_has_no_ones.any():
    # for those rows, force one random position to 1
    rand_idx = torch.randint(0, seq_len, (row_has_no_ones.sum(),))
    mask[row_has_no_ones, rand_idx] = 1

y = (X * mask.unsqueeze(-1)).sum(dim=-2)

epochs = 100
batch_size = 100
model = Model(embed_dim=embed_dim, num_head=4)
optimizer = torch.optim.Adam(model.parameters())
criteria = nn.MSELoss()
for epoch in range(epochs):
  for i in range(0, n, batch_size):
    model.train()
    end = i + batch_size
    batch_X = X[i:end]
    batch_mask = mask[i:end]
    batch_y = y[i:end]
    attention_mask = batch_mask[:, None, None, :].expand(batch_mask.shape[0], 1, batch_mask.shape[1], batch_mask.shape[1]).float()
    attention_mask = attention_mask.masked_fill(attention_mask == 0, float("-inf"))
    logits = model(batch_X, attention_mask)
    loss = criteria(logits, batch_y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if i == n // batch_size:
      print(f"Epoch {epoch} loss {loss.item()}")