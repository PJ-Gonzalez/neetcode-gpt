import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        stats = []

        def get_leaf_layers(module):
            children = list(module.children())
            if not children:
                yield module
            else:
                for child in children:
                    yield from get_leaf_layers(child)

        layers = list(get_leaf_layers(model))

        with torch.no_grad():
            output = x
            for i, layer in enumerate(layers):
                output = layer(output)

                if isinstance(layer, nn.Linear):
                    dead_fraction = 0.0

                    if i + 1 < len(layers) and isinstance(layers[i + 1], nn.ReLU):
                        relu_output = torch.relu(output)
                        zero_mask = relu_output == 0

                        if zero_mask.ndim > 1:
                            dead_neurons = zero_mask.flatten(0, -2).all(dim=0)
                            dead_fraction = float(dead_neurons.float().mean())
                        else:
                            dead_fraction = float(zero_mask.float().mean())

                    stats.append({
                        "mean": round(float(output.mean()), 4),
                        "std": round(float(output.std()), 4),
                        "dead_fraction": round(dead_fraction, 4),
                    })

        return stats

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()
        prediction = model(x)
        loss = nn.MSELoss()(prediction, y)
        loss.backward()

        stats = []
        for layer in model.modules():
            if isinstance(layer, nn.Linear) and layer.weight.grad is not None:
                grad = layer.weight.grad
                stats.append({
                    "mean": round(float(grad.mean()), 4),
                    "std": round(float(grad.std()), 4),
                    "norm": round(float(grad.norm()), 4),
                })

        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        if any(s["dead_fraction"] > 0.5 for s in activation_stats):
            return "dead_neurons"

        norms = [s["norm"] for s in gradient_stats]
        if any(norm > 100 for norm in norms):
            return "exploding_gradients"
        if any(norm < 1e-6 for norm in norms):
            return "vanishing_gradients"

        return "healthy"
