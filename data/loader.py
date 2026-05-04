import torch
from torchtyping import TensorType
from typing import Tuple

class Solution:
    def create_batches(self, data: TensorType[int], context_length: int, batch_size: int) -> Tuple[TensorType[int], TensorType[int]]:
        # data: 1D tensor of encoded text (integer token IDs)
        # context_length: number of tokens in each training example
        # batch_size: number of examples per batch
        #
        # Return (X, Y) where:
        # - X has shape (batch_size, context_length)
        # - Y has shape (batch_size, context_length)
        # - Y is X shifted right by 1 (Y[i][j] = data[start_i + j + 1])
        #
        # Use torch.manual_seed(0) before generating random start indices
        # Use torch.randint to pick random starting positions
        torch.manual_seed(0)
        rand = torch.randint(data.shape[0] - context_length, (batch_size, ))
        X = torch.zeros(batch_size, context_length, dtype=data.dtype)  
        Y = torch.zeros(batch_size, context_length, dtype=data.dtype)  

        for i, start in enumerate(rand):
            X[i] = data[start : start + context_length]
            Y[i] = data[start + 1 : start + 1 + context_length]

        return (X, Y)