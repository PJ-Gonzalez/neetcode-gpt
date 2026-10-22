import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        # Forward pass through the model.
        # After each ReLU layer, compute the fraction of neurons that are dead.
        # A neuron is dead if it outputs 0 for ALL samples in the batch.
        # Return a list of dead fractions (one per ReLU layer), rounded to 4 decimals.
        def get_leaf_layers(module):
            children = list(module.children())
            if not children:
                yield module
            else:
                for child in children:
                    yield from get_leaf_layers(child)

        dead_fractions = []

        with torch.no_grad():
            output = x
            for layer in get_leaf_layers(model):
                output = layer(output)

                if isinstance(layer, nn.ReLU):
                    zero_mask = output == 0
                    if output.ndim > 1:
                        dead_neurons = zero_mask.flatten(0, -2).all(dim=0)
                        fraction = dead_neurons.float().mean()
                    else:
                        fraction = zero_mask.float().mean()

                    dead_fractions.append(fraction)

        if not dead_fractions:
            return []

        return [
            round(value, 4)
            for value in torch.stack(dead_fractions).cpu().tolist()
        ]

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        # Given dead fractions per ReLU layer, suggest a fix.
        # Check in this order:
        # 1. 'use_leaky_relu' if any layer has dead fraction > 0.5
        # 2. 'reinitialize' if the first layer has dead fraction > 0.3
        # 3. 'reduce_learning_rate' if dead fraction strictly increases
        #    with depth AND the last layer's fraction > 0.1
        # 4. 'healthy' if max dead fraction < 0.1
        # 5. 'healthy' otherwise
        if any(fraction > 0.5 for fraction in dead_fractions):
            return "use_leaky_relu"

        if dead_fractions and dead_fractions[0] > 0.3:
            return "reinitialize"

        if (
            len(dead_fractions) >= 2
            and all(
                earlier < later
                for earlier, later in zip(dead_fractions, dead_fractions[1:])
            )
            and dead_fractions[-1] > 0.1
        ):
            return "reduce_learning_rate"

        if not dead_fractions or max(dead_fractions) < 0.1:
            return "healthy"

        return "healthy"
