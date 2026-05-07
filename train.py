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
        N = data.shape[0]
        vocab_size = data.max().item() + 1
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        for epochs in range(epochs):
            torch.manual_seed(epochs)
            rand = torch.randint( N-context_length, (batch_size, ) )
            X = torch.stack( [data[start:start+context_length] for start in rand] )
            Y = torch.stack( [data[start+1:start+context_length+1] for start in rand] )

            logits = model(X)
            logits = logits.reshape(batch_size * context_length, vocab_size)
            Y = Y.reshape(batch_size * context_length)

            loss = F.cross_entropy(logits, Y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return round(loss.item(), 4)