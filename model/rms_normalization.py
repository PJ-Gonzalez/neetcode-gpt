import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        x_arr = np.asarray(x, dtype=float)
        gamma_arr = np.asarray(gamma, dtype=float)

        normalized = x_arr / np.sqrt(np.mean(x_arr * x_arr) + eps)
        return np.round(gamma_arr * normalized, 4).tolist()
