import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, mask=None):
        d_k = q.size(-1)
        scores = (
            q
            @ torch.transpose(k, -2, -1)
            / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        )
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = torch.softmax(scores, dim=-1)

        result = attention_weights @ v
        return result, attention_weights
