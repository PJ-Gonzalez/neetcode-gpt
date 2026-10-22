import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        def get_leaf_layers(module):
            children = list(module.children())
            if not children:
                yield module
            else:
                for child in children:
                    yield from get_leaf_layers(child)

        layers = list(get_leaf_layers(model))
        values = []

        with torch.no_grad():
            output = x
            for i, layer in enumerate(layers):
                output = layer(output)

                if isinstance(layer, nn.Linear):
                    dead_fraction = output.new_zeros(())

                    if i + 1 < len(layers) and isinstance(layers[i + 1], nn.ReLU):
                        non_positive = output <= 0
                        if output.ndim > 1:
                            dead_neurons = non_positive.flatten(0, -2).all(dim=0)
                            dead_fraction = dead_neurons.float().mean()
                        else:
                            dead_fraction = non_positive.float().mean()

                    values.append(torch.stack((
                        output.mean(),
                        output.std(),
                        dead_fraction,
                    )))

        if not values:
            return []

        rows = torch.stack(values).cpu().tolist()
        return [
            {
                "mean": round(mean, 4),
                "std": round(std, 4),
                "dead_fraction": round(dead_fraction, 4),
            }
            for mean, std, dead_fraction in rows
        ]

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad(set_to_none=True)

        prediction = model(x)
        loss = nn.functional.mse_loss(prediction, y)
        loss.backward()
        del loss, prediction

        values = []
        for layer in model.modules():
            if isinstance(layer, nn.Linear) and layer.weight.grad is not None:
                grad = layer.weight.grad
                values.append(torch.stack((
                    grad.mean(),
                    grad.std(),
                    grad.norm(),
                )))

        if not values:
            return []

        rows = torch.stack(values).cpu().tolist()
        return [
            {
                "mean": round(mean, 4),
                "std": round(std, 4),
                "norm": round(norm, 4),
            }
            for mean, std, norm in rows
        ]

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        if any(s["dead_fraction"] > 0.5 for s in activation_stats):
            return "dead_neurons"
        if any(s["norm"] > 100 for s in gradient_stats):
            return "exploding_gradients"
        if any(s["norm"] < 1e-6 for s in gradient_stats):
            return "vanishing_gradients"
        return "healthy"
