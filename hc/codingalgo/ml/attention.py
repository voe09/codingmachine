import torch
import torch.nn.functional as F

from torch import nn
from torch import optim

class SelfAttention(nn.Module):

  def __init__(self, emb_dim: int, num_head: int):
    super().__init__()
    self.emb_dim = emb_dim
    self.num_head = num_head
    self.head_dim = emb_dim // num_head

    assert self.head_dim * self.num_head == self.emb_dim

    self.q_proj = nn.Linear(self.emb_dim, self.emb_dim)
    self.k_proj = nn.Linear(self.emb_dim, self.emb_dim)
    self.v_proj = nn.Linear(self.emb_dim, self.emb_dim)
    self.o_proj = nn.Linear(self.emb_dim, self.emb_dim)

  def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    B, S, _ = x.shape
    q = self.q_proj(x) # B, S, D
    k = self.k_proj(x) # B, S, D
    v = self.v_proj(x) # B, S, D

    q = q.view((B, S, self.num_head, self.head_dim)).transpose(1, 2) # B, num_head, S, head_dim
    k = k.view((B, S, self.num_head, self.head_dim)).transpose(1, 2)
    v = v.view((B, S, self.num_head, self.head_dim)).transpose(1, 2)

    scaling = self.head_dim ** 0.5
    # (B, num_head, S, head_dim) @ (B, num_head, head_dim, S) -> (B, num_head, S, S)
    attention_score = torch.matmul(q, k.transpose(-2, -1)) / scaling

    if mask != None:
      attention_score.masked_fill_(mask == 0, float('-inf'))

    attention_score = F.softmax(attention_score, dim=-1)
    out = attention_score @ v # B, num_head, S, head_dim
    out = out.transpose(1, 2).contiguous().view((B, S, self.emb_dim))
    out = self.o_proj(out)
    return out
  

# Test
seq = 10
emb_dim = 512
num_heads = 8
batch_size = 1000

model = SelfAttention(emb_dim, num_heads)

input_data = torch.randn(batch_size, seq, emb_dim)
for i in range(2, seq):
  input_data[:, i, :] = 0.8 * input_data[:, i - 2, :] + 0.2 * input_data[:, i-1, :]

target_data = input_data.clone()

criteria = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
  causal_strict = torch.tril(
    torch.ones((seq, seq), dtype=torch.bool, device=input_data.device),
    diagonal=-1
  )
  causal_strict[0,0] = 1  # special case: let token 0 attend to itself
  mask = causal_strict.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, seq, seq)  
  outputs = model(input_data[2:], mask[2:])
  loss = criteria(outputs, target_data[2:])

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  if (epoch + 1) % 10 == 0:
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')



# 09/29 - simple self attn
import torch
import torch.nn.functional as F

from torch import nn

class SelfAttention(nn.Module):

  def __init__(self, emb_dim: int):
    super().__init__()
    # WQ dxd, WK dxd WV dxd
    self.q_proj = nn.Linear(emb_dim, emb_dim)
    self.k_proj = nn.Linear(emb_dim, emb_dim)
    self.v_proj = nn.Linear(emb_dim, emb_dim)
    self.o_proj = nn.Linear(emb_dim, emb_dim)
    # Original impl
    """
    self.o_proj = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.SiLU())
    acts like a linear layer but not necessary and is not the standard
    """

    self.d = emb_dim
    self.scaling = 1 / (emb_dim ** 0.5)

    ## original impl
    """
    self.d = torch.tensor(emb_dim)
    self.scaling = 1 / torch.sqrt(self.d) -> easy to lead tensor device mismatch
    """

  def forward(self, X: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    # X: (b, S, d)
    Q = self.q_proj(X) # b, S, d
    K = self.k_proj(X) # b, S, d
    V = self.v_proj(X) # b, S, d

    # attn weights, softmax(QK.T/sqrt(d))
    attn_weights = Q @ K.transpose(-2, -1) * self.scaling # (b, S, S)
    if mask is not None: # original: not implemented
      attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))
    attn_weights = F.softmax(attn_weights, dim=-1) # (b, S, S)
    # original impl attn_weights = F.softmax(attn_weights) # (b, S, S) -> while it apply along the last dim which is right, it might be perform incorrectly when we have multi head attn

    attn_values = attn_weights @ V # (b, S, d)

    outputs = self.o_proj(attn_values)

    return outputs
  
n = 1000
seqLen = 10
emb_dim = 10
eps = 1e-3

data = torch.randn(n, seqLen, emb_dim)
for i in range(3, seqLen):
  data[:, i, :] = 0.2 * data[:, i-2, :] + 0.8 * data[:, i - 1, :] + eps


targets = data.clone()

model = SelfAttention(emb_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criteria = nn.MSELoss()

epochs = 100
batch_size = 500
step = 0
print_per_step = 10

for epoch in range(epochs):
  for i in range(0, n, batch_size):
    model.train()
    X = data[i:i+batch_size, :, :]
    y = model(X)[:, 2:, :]
    label = targets[i:i+batch_size, 2:, :]
    loss = criteria(label, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if step % print_per_step == 0:
      print(f"Epoch {epoch} loss {loss.item()}")
    step += 1

"""
Issue of the original impl -> slow convergence
"""