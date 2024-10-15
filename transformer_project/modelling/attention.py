import torch
from torch import nn
import math


class Attention(nn.Module):
    def __init__(self, mask_future=False):
        super().__init__()
        self.mask_future = mask_future

    def forward(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)

        # add asserts for dimension matching

        batch_size = query.size(0)
        seq_length_q = query.size(-2)
        seq_length_k = key.size(-2)

        if self.mask_future:
            future_tokens_mask = torch.triu(
                torch.ones(seq_length_q, seq_length_k), diagonal=1
            )
            scores = scores.masked_fill(future_tokens_mask==1,-1e9)

        if mask is not None:
            padding_mask = mask.unsqueeze(1).expand(batch_size, seq_length_q, seq_length_k)
            scores = scores.masked_fill(padding_mask==0, -1e9)

        attention_weights = torch.softmax(scores, dim=-1)

        result = attention_weights @ value
        return result