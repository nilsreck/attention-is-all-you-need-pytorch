import torch
from torch import nn
from modelling.attention import BaseTransformerLayer, TransformerDecoderLayer
from modelling.huggingface_bpe_tokenizer import CustomTokenizer
from modelling.positional_encoding import PositonalEncoding


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        n_heads,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        dropout,
        maxlen,
    ):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim=d_model)
        self.positional_encoding = PositonalEncoding(d_model, maxlen)
        self.encoder_layers = nn.ModuleList(
            [
                BaseTransformerLayer(d_model, n_heads, dim_feedforward, dropout)
                for _ in range(num_encoder_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(d_model, n_heads, dim_feedforward, dropout)
                for _ in range(num_decoder_layers)
            ]
        )

    def forward(self, input, output, attention_mask=None, encoder_attention_mask=None):
        input = self.embedding_layer(input) + self.positional_encoding(input)
        mem = input
        for layer in self.encoder_layers:
            mem = layer(input, attention_mask)

        output = self.embedding_layer(output) + self.positional_encoding(output)
        for layer in self.decoder_layers:
            output = layer(output, mem, encoder_attention_mask)

        return output
