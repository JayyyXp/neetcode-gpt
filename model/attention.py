import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.key = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        Q = self.query(embedded)
        K = self.key(embedded)
        V = self.value(embedded)

        d_k = K.shape[2]

        inside = (Q @ K.transpose(1, 2)) / (d_k ** 0.5)
        
        lower_tri = torch.tril(torch.ones(embedded.shape[1], embedded.shape[1]))
        mask = lower_tri == 0
        pre_soft = inside.masked_fill(mask, -float('inf'))
        
        z = pre_soft.softmax(dim=2) @ V

        return torch.round(z, decimals=4)

