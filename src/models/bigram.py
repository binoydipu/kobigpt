import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    """
    A simple bigram language model that predicts the next token based on the current token.
    This model uses an embedding layer to map each token to a vector of logits for the next
    token in the vocabulary.

    Args:
        vocab_size (int): The size of the vocabulary.
    """

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        # idx is token no, it takes out idx-th row from the table (C)
        # it contains logits of all other tokens occurs after idx-th token
        logits = self.token_embedding_table(idx) # (B,T,C) -> (batch_size, block_size, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape

            # every B*T contains a token for which we want to run prediction 
            logits = logits.view(B*T, C) # 2-dim 
            targets = targets.view(B*T)  # 1-dim
            
            # cross_entropy need (B,C,T) Tensor
            loss = F.cross_entropy(logits, targets) # calculate error of the prediction

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        # for all B (batch dims), generate tokens for T (time) dims
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            # only last item of T (time) dim predicts what comes next
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax across C (total token) dim to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution, for each batch we predict 1 token
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
