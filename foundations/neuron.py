import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)
        z = float(np.dot(x, w) + b)

        if activation == "sigmoid":
            y = 1 / (1 + np.exp(-z))
        elif activation == "relu":
            y = np.maximum(0, z)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        return round(float(y), 5)
