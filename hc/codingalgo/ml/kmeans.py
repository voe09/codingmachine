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