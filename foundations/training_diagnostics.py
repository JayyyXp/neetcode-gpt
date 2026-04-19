import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        ans = []
        with torch.no_grad():
            prev = x
            for layer in model:
                prev = layer(prev)
                if not isinstance(layer, nn.Linear): 
                    continue 
                mean = prev.mean()
                std = prev.std()
                dead_fraction = ((prev <= 0).all(dim=0)).float().mean()
                
                ans.append(
                    {
                        "mean": round(mean.item(), 4),
                        "std": round(std.item(), 4),
                        "dead_fraction": round(dead_fraction.item(), 4)
                    }
                )
        return ans

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()
        y_hat = model.forward(x)
        loss = nn.MSELoss()(y_hat, y)
        loss.backward()
        ans = []
        for layer in model:
            if not isinstance(layer, nn.Linear): 
                continue 
            w = layer.weight.grad
            mean = w.mean()
            std = w.std()
            norm = torch.norm(w)
            
            ans.append(
                {
                    "mean": round(mean.item(), 4),
                    "std": round(std.item(), 4),
                    "norm": round(norm.item(), 4)
                }
            )
        return ans
    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        
        if any(item["dead_fraction"] > 0.5 for item in activation_stats):
            return "dead_neurons"
        
        if any(item["norm"] > 1_000 for item in gradient_stats):
            return "exploding_gradients"

        if gradient_stats[-1]["norm"] < 1e-5:
            return "vanishing_gradients"

        for item in activation_stats:  
            if item["std"] < 0.1:
                return "vanishing_gradients"
            if item["std"] > 10.0:
                return "exploding_gradients"

        return "healthy"
