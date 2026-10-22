import torch
import torch.nn as nn
import torch.nn.functional as F

# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        # Train the GPT model using AdamW and cross_entropy loss.
        # For each epoch: seed with torch.manual_seed(epoch),
        # sample batches from data, run forward/backward, update weights.
        # Return the final loss rounded to 4 decimals.
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        windows = data.unfold(0, context_length + 1, 1)
        model.train()

        final_loss = None

        for epoch in range(epochs):
            torch.manual_seed(epoch)
            starts = torch.randint(
                len(data) - context_length,
                (batch_size,),
                device=data.device,
            )

            batch = windows[starts]
            x = batch[:, :-1]
            y = batch[:, 1:]

            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )
            loss.backward()
            optimizer.step()

            final_loss = loss.item()

        return round(final_loss, 4)
