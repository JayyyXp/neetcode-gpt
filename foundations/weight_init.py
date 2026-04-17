import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std = math.sqrt(2 / (fan_in + fan_out))
        tensor = torch.randn(fan_out, fan_in) * std
        return [[round(v.item(), 4) for v in row] for row in tensor] 

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std = math.sqrt(2 / (fan_in))
        tensor =  torch.randn(fan_out, fan_in) * std
        return [[round(v.item(), 4) for v in row] for row in tensor]
        #return [
        #    [random.gauss(mu=0, sigma=std)] * fan_in
        #    for _ in range(fan_out)
        #]

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.
        torch.manual_seed(0)
        weights = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            if init_type == "xavier":
                std = math.sqrt(2 / (in_dim + hidden_dim))
            elif init_type == "kaiming":
                std = math.sqrt(2 / in_dim)
            else:
                std = 1.0

            w = torch.randn(in_dim, hidden_dim) * std
            weights.append(w)
        ans = []
        prev = torch.randn(input_dim)
        for w in weights:
            a = nn.functional.relu(w @ prev)
            ans.append(round(a.std().item(), 2))
            prev = a
        return ans