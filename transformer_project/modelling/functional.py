import torch.nn as nn
import os
import sys
from transformer_project.modelling.attention import MultiHeadAttention
from transformer_project.modelling.ffn import PositionWiseFeedForward

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BaseTransformerLayer(nn.Module):
    def __init__(self, input_dim, num_heads, feature_dim, dropout=0.0):
        super().__init__()
        self.input_dim = input_dim

        self.self_attention = MultiHeadAttention(input_dim, num_heads)
        self.layer_norm_1 = nn.LayerNorm(input_dim)
        self.feature_transformation = PositionWiseFeedForward(input_dim, feature_dim)
        self.layer_norm_2 = nn.LayerNorm(input_dim)

    def forward(self, input, attention_mask):
        attention_output = self.self_attention(input, input, input, attention_mask)
        residual_connections_mha = attention_output + input
        normalized_mha_output = self.layer_norm_1(residual_connections_mha)

        ffn_outputs = self.feature_transformation(normalized_mha_output)
        residual_connections_ffn = ffn_outputs + normalized_mha_output
        final = self.layer_norm_2(residual_connections_ffn)

        return final


class TransformerDecoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, feature_dim, dropout=0.0):
        super().__init__()

        self.input_dim = input_dim

        self.self_attention = MultiHeadAttention(input_dim, num_heads, mask_future=True)
        self.layer_norm_1 = nn.LayerNorm(input_dim)
        self.encoder_attention = MultiHeadAttention(input_dim, num_heads)
        self.layer_norm_2 = nn.LayerNorm(input_dim)
        self.feature_transformation = PositionWiseFeedForward(input_dim, feature_dim)
        self.layer_norm_3 = nn.LayerNorm(input_dim)

    def forward(self, input, encoder_output, encoder_attention_mask, attention_mask):
        masked_attention_output = self.self_attention(
            input, input, input, attention_mask
        )
        residual_connections_mha = masked_attention_output + input
        normalized_queries = self.layer_norm_1(residual_connections_mha)

        attention_output = self.encoder_attention(
            normalized_queries,
            encoder_output,
            encoder_output,
            encoder_attention_mask,
        )
        residual_connections_mha2 = attention_output + normalized_queries
        normalized_mha_output2 = self.layer_norm_2(residual_connections_mha2)

        ffn_outputs = self.feature_transformation(normalized_mha_output2)
        residual_connections_ffn = ffn_outputs + normalized_mha_output2
        final = self.layer_norm_3(residual_connections_ffn)

        return final
