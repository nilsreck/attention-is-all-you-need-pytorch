import os
import sys
import torch
from torch import nn
import math

# Add the parent directory to the system path for importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
            ).to(query.device)
            # -1e4 to comply with mixed precision
            scores = scores.masked_fill(future_tokens_mask == 1, -1e4)

        if mask is not None:
            padding_mask = (
                mask.unsqueeze(1)
                .expand(batch_size, seq_length_q, seq_length_k)
                .to(query.device)
            )
            # -1e4 to comply with mixed precision
            scores = scores.masked_fill(padding_mask == 0, -1e4)

        attention_weights = torch.softmax(scores, dim=-1)

        result = attention_weights @ value
        return result


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, mask_future=False):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.mask_future = mask_future

        self.query_transform = nn.Linear(d_model, d_model, bias=False)
        self.key_transform = nn.Linear(d_model, d_model, bias=False)
        self.value_transform = nn.Linear(d_model, d_model, bias=False)
        self.output_transform = nn.Linear(d_model, d_model, bias=False)

        self.attention = Attention(mask_future=self.mask_future)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_length_q = query.size(-2)
        seq_length_k = key.size(-2)
        seq_length_v = value.size(-2)

        # test dims: query = (2,3,4)
        query = self.query_transform(query)  # (2,3,4) x (4,4) <- broadcast
        # dims after linear transform: (2,3,4)
        key = self.key_transform(key)
        value = self.value_transform(value)

        # Split embedddings in half (for num_heads = 2), then reshape to put heads dim ahead of sequence length dim
        query = torch.einsum(
            "bqhd->bhqd", query.view(batch_size, seq_length_q, self.num_heads, self.d_k)
        )
        key = torch.einsum(
            "bqhd->bhqd", key.view(batch_size, seq_length_k, self.num_heads, self.d_k)
        )
        value = torch.einsum(
            "bqhd->bhqd", value.view(batch_size, seq_length_v, self.num_heads, self.d_k)
        )

        # Transform into
        # [
        # 4 (one for each head for each batch),
        # 3 (seq len),
        # 2 (dim per head)
        # ]:
        query = query.reshape(batch_size * self.num_heads, seq_length_q, self.d_k)
        key = key.reshape(batch_size * self.num_heads, seq_length_k, self.d_k)
        value = value.reshape(batch_size * self.num_heads, seq_length_v, self.d_k)

        if mask is not None:
            mask = mask.unsqueeze(1).expand(batch_size, self.num_heads, seq_length_k)
            mask = mask.reshape(batch_size * self.num_heads, seq_length_k)

        attention_output = self.attention(query, key, value, mask)

        # Reshape to original shape
        attention_output = attention_output.view(
            batch_size, self.num_heads, seq_length_q, self.d_k
        )

        # Concatenate heads, apply final linear transformation
        attention_output = torch.einsum("bhqd->bqhd", attention_output).reshape(
            batch_size, seq_length_q, self.num_heads * self.d_k
        )

        output = self.output_transform(attention_output)
        return output
