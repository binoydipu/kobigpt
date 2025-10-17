import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 256    # context length
    vocab_size: int = None        # the size of the vocabulary
    n_layer: int = 6        # number of layers
    n_head: int = 6         # number of attention heads
    n_embd: int = 384       # token embedding dimension
    dropout: float = 0.0    # dropout rate
    bias: bool = True       # use bias in the Linear & Norm layers



class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, config, is_causal=False):
        super().__init__()
        self.head_size = config.n_embd // config.n_head

        self.is_causal = is_causal  # if True, apply causal mask to ensure that attention is only applied to the left in the input sequence
        self.key = nn.Linear(config.n_embd, self.head_size, bias=False)
        self.query = nn.Linear(config.n_embd, self.head_size, bias=False)
        self.value = nn.Linear(config.n_embd, self.head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)

        # compute attention scores ("affinities")
        # dk**-0.5 is scaled dot-product attention, helps with stability
        dk = k.size(-1)
        att = q @ k.transpose(-2, -1) * dk**-0.5 # (B,T,T)
        if self.is_causal:
            att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
        att = F.softmax(att, dim=-1) # (B,T,T)
        
        att = self.dropout(att)
        v = self.value(x) # (B,T,head_size)

        out = att @ v # (B,T,head_size)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, config, is_causal=False):
        super().__init__()

        self.heads = nn.ModuleList([Head(config, is_causal) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # concatenate the output of all heads
        out = torch.cat([h(x) for h in self.heads], dim=-1) # concatenate the output of all heads
        
        # project helps to mix the information from different heads
        out = self.proj(out) # project back to the original embedding dimension
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, config):
        super().__init__()

        self.net = nn.Sequential(
            
            # Gives model capacity to represent richer nonlinear interactions per token
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(), # Gaussian Error Linear Unit activation function
            
            # Ensures the output has the same shape as the input so it can be added back residually
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, config, is_causal=False):
        super().__init__()

        self.attn = MultiHeadAttention(config, is_causal)
        self.ffwd = FeedForward(config)
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)

    def forward(self, x):
        # add residual connections around the two sub-layers
        x = x + self.attn(self.ln_1(x))     # apply layer norm before self-attention
        x = x + self.ffwd(self.ln_2(x))     # apply layer norm before feed-forward
        return x


class KobiGPTModel(nn.Module):
    """ the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(
            *[Block(config, is_causal=True) for _ in range(config.n_layer)],
        )
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

        self.apply(self._init_weights) 

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # ----------- Input Embedding + Positional Encoding -----------

        # for each index in idx, get the corresponding token embedding
        token_embd = self.token_embedding_table(idx) # (B,T,n_embd)
        
        # positional embeddings for each position in the sequence
        pos_embd = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,n_embd)
        
        # now add the two embeddings together to get the final token representation
        # x now has both token identity and positional information
        x = token_embd + pos_embd # (B,T,n_embd)
        # -------------------------------------------------------------

        
        # ----------- Forward the input through the Transformer -----------

        # pass the input through the series of Transformer blocks 
        # each block contains self-attention and feed-forward layers
        # final x has contextualized token representations
        x = self.blocks(x) # (B,T,n_embd)
        
        # final layer norm to stabilize and normalize the output
        x = self.ln_f(x) # (B,T,n_embd)
        # ---------------------------------------------------------------


        # ----------- Output of the Language Model ---------------------
        
        # project the final hidden states to the vocabulary size to get logits for each token
        logits = self.lm_head(x) # (B,T,vocab_size)
        # ---------------------------------------------------------------


        if targets is None: # during inference, we only have idx and no targets
            loss = None
        else:
            B, T, C = logits.shape

            # every B*T contains a token for which we want to run prediction 
            logits = logits.view(B*T, C) # 2-dim 
            targets = targets.view(B*T)  # 1-dim
            
            # cross_entropy need (B,C,T) Tensor
            loss = F.cross_entropy(logits, targets) # calculate error of the prediction

        return logits, loss

    def generate(self, idx, max_new_tokens, temparature=1.0):
        """
        Generate new tokens given a context idx. Tweak the temperature to control randomness.

        Args:
            idx: (B, T) array of indices in the current context
            max_new_tokens: number of tokens to generate
            temperature: float value to modulate the next token probabilities
        Returns:
            idx: (B, T + max_new_tokens) array of indices in the extended context
        Example:
            >>> context = torch.zeros((1, 1), dtype=torch.long) # starting token
            >>> generated = model.generate(context, max_new_tokens=100)
            >>> print(decode(generated[0].tolist()))
        """

        # idx is (B, T) array of indices in the current context
        # for all B (batch dims), generate tokens for T (time) dims
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.config.block_size:] # (B, block_size)
            
            # get the predictions
            logits, loss = self(idx_cond)
            
            # focus only on the last time step
            # only last item of T (time) dim predicts what comes next
            logits = logits[:, -1, :] # becomes (B, C)
            
            # apply softmax across C (total token) dim to get probabilities
            probs = F.softmax(logits / temparature, dim=-1) # (B, C)
            
            # sample from the distribution, for each batch we predict 1 token
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

            # Debug: print progress every 100 tokens
            if _ % 100 == 0:
                print(f"Currently generated {_} tokens...")
        
        return idx
