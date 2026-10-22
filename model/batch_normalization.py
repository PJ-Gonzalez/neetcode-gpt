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
        x_arr = np.asarray(x, dtype=float)
        gamma_arr = np.asarray(gamma, dtype=float)
        beta_arr = np.asarray(beta, dtype=float)
        running_mean_arr = np.asarray(running_mean, dtype=float)
        running_var_arr = np.asarray(running_var, dtype=float)

        if training:
            mean = np.mean(x_arr, axis=0)
            var = np.var(x_arr, axis=0)

            running_mean_arr = (1 - momentum) * running_mean_arr + momentum * mean
            running_var_arr = (1 - momentum) * running_var_arr + momentum * var
        else:
            mean = running_mean_arr
            var = running_var_arr

        x_hat = (x_arr - mean) / np.sqrt(var + eps)
        y = gamma_arr * x_hat + beta_arr

        return (
            np.round(y, 4).tolist(),
            np.round(running_mean_arr, 4).tolist(),
            np.round(running_var_arr, 4).tolist(),
        )
