import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, padding_mask):
        d_k = q.size(-1)
        scores = (
            q
            @ torch.transpose(k, -2, -1)
            / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        )

        combined_mask = self.generate_combined_mask(q, padding_mask)

        # if padding_mask is not None:
        scores = scores.masked_fill(combined_mask == 1, -1e9)

        attention_weights = torch.softmax(scores, dim=-1)

        result = attention_weights @ v
        return result, attention_weights

    def generate_combined_mask(self, q, padding_mask):
        seq_length = q.size(-2)

        future_tokens_mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1)

        combined_mask = future_tokens_mask + padding_mask
        return combined_mask
