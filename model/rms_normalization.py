import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        N = len(x)
        RMS = ((1 / N) * sum([xi**2 for xi in x]) + eps) ** 0.5
        x_hat = [xi / RMS for xi in x]
        return [round(x_hat_i * gamma_i, 4) for x_hat_i, gamma_i in zip(x_hat, gamma)]