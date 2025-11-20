# model.py
import torch
import torch.nn as nn


class TinyCharRNN(nn.Module):
    """
    A very small GRU-based character-level language model.
    Good enough for demos / CI and fast on CPU.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 128, num_layers: int = 1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        """
        x: [B, T] of token indices
        hidden: optional hidden state
        """
        emb = self.embed(x)          # [B, T, E]
        out, hidden = self.rnn(emb, hidden)  # out: [B, T, H]
        logits = self.fc(out)        # [B, T, V]
        return logits, hidden
