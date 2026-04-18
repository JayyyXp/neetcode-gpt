import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        batch_size = len(x)
        N = len(x[0])
        if training:
            # averages across batch for each feature
            mu_B = [
                (1/batch_size) * sum(x[i][j] for i in range(batch_size))
                for j in range(N)
            ]
            
            sigma_B = [
                (1/batch_size) * sum((x[i][j] - mu_B[j])**2 for i in range(batch_size))
                for j in range(N)
            ]
            
            x_hat = [
                [(x[i][j] - mu_B[j]) / (sigma_B[j] + eps)**0.5 for j in range(N)]
                for i in range(batch_size)
            ]

            for i in range(len(running_var)):
                running_mean[i] = round((1-momentum) * running_mean[i] + momentum * mu_B[i], 4)
            for i in range(len(running_var)):
                running_var[i] = round((1-momentum) * running_var[i] + momentum * sigma_B[i], 4)
        else:
            x_hat = [
                [(x[i][j] - running_mean[j]) / (running_var[j] + eps)**0.5 for j in range(N)]
                for i in range(batch_size)
            ]

        y = [
            [round(gamma[j] * x_hat[i][j] + beta[j], 4) for j in range(N)]
            for i in range(batch_size)
        ]

        return (
            y,
            running_mean,
            running_var
        )
