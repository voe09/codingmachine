import torch

import torch.nn.functional as F

class KMeans:
    def __init__(self, k: int, num_iter: int = 1000, tol: float = 1e-4):
        self.k = k
        self.num_iter = num_iter
        self.tol = tol
        self.centroid = None

    def fit(self, x: torch.Tensor):
        bs, D = x.shape
        # initialize centroid
        init_idx = torch.randperm(bs)[: self.k]
        self.centroid = x[init_idx, :]

        for i in range(self.num_iter):
            distances = cosine_distance(x, self.centroid) # (bs, D) x (k, D) -> (bs, K)
            
            cluster_ids = torch.argmax(distances, dim = 1) # (bs, 1)

            new_centroid = torch.zeros_like(self.centroid)
            for j in range(self.k):
                points_in_centroid = x[cluster_ids == j]
                if len(points_in_centroid) > 0:
                    new_centroid[j] = points_in_centroid.mean(dim=0)
                else:
                    new_centroid[j] = x[torch.randint(0, bs, (1,))]
            
            centroid_shift = torch.norm(self.centroid - new_centroid, dim=1).sum()
            if centroid_shift < self.tol:
                break
            
            self.centroid = new_centroid
        
    def predict(self, x: torch.Tensor):
        distances = cosine_distance(x, self.centroid)
        cluster_ids = torch.argmax(distances, dim=1)
        return cluster_ids
    


def cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_norm = F.normalize(x, p = 2, dim = 1)
    y_norm = F.normalize(y, p = 2, dim = 1)
    distance = 1 - x_norm @ y_norm
    return distance


def test_kmeans_simple_clusters():
    cluster1 = torch.randn(10, 2) + torch.tensor([5.0, 5.0])
    cluster2 = torch.randn(10, 2) + torch.tensor([-5.0, -5.0])
    x = torch.cat([cluster1, cluster2], dim = 0)

    kmeans = KMeans(k = 2)
    kmeans.fit(x)
    preds = kmeans.predict(x)
    print(preds)

test_kmeans_simple_clusters()




import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim


class KMeans(nn.Module):

  def __init__(self, k: int, dim: int):
    super().__init__()
    self.k = k
    self.centroids = nn.Parameter(torch.randn(k, dim)) # K, dim

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    distances = self.distance(inputs) # B, k
    indices = distances.argmin(dim=1)
    return indices

  def distance(self, inputs: torch.Tensor) -> torch.Tensor:
    sim = F.normalize(inputs, dim=1) @ F.normalize(self.centroids, dim=1).T
    return 1 - sim

  def loss(self, inputs: torch.Tensor) -> torch.Tensor:
    distances = self.distance(inputs)
    min_dist, _ = distances.min(dim=1)
    return min_dist.mean()
  

  def train(k: int, data: torch.Tensor, epochs: int, batch_size: int):
  n, dim = data.shape
  model = KMeans(k, dim)
  optimizer = optim.Adam(model.parameters(), lr=0.1)

  for epoch in range(epochs):
    perm = torch.randperm(n)
    for i in range(0, n, batch_size):
      optimizer.zero_grad()
      batch = data[perm[i:i+batch_size]]
      loss = model.loss(batch)
      loss.backward()
      optimizer.step()

    print(f"epoch {epoch}, loss: {loss.item()}")
    print(f"epoch {epoch}, centroids: {model.centroids}")

  return model

cluster1 = torch.randn(1000, 2) + torch.tensor([5.0, 5.0])
cluster2 = torch.randn(1000, 2) + torch.tensor([-5.0, -5.0])
x = torch.cat([cluster1, cluster2], dim = 0)

model = train(2, x, 10, 100)