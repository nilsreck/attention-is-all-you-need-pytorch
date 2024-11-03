import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, sq_len):
        super().__init__()
        position = torch.arange(sq_len, dtype=torch.float).unsqueeze(1)
        div_term = 10000 ** (torch.arange(0, embedding_dim, 2) / embedding_dim)
        pe = torch.zeros(sq_len, embedding_dim)
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)
        self.register_buffer("positional_embedding", pe)

    def forward(self, input):
        return input + self.positional_embedding[: input.size(1)]
