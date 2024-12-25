from torch import nn

from transformer_project.modelling.functional import (
    BaseTransformerLayer,
    TransformerDecoderLayer,
)
from transformer_project.modelling.positional_encoding import PositionalEncoding


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        n_heads,
        num_encoder_layers,
        num_decoder_layers,
        dim_feed_forward,
        dropout,
        maxlen,
    ):
        super().__init__()
        self.embedding_layer = nn.Embedding(
            vocab_size, embedding_dim=d_model
        )  # shared embeddings
        self.positional_encoding = PositionalEncoding(
            embedding_dim=d_model, sq_len=maxlen
        )
        self.encoder_layers = nn.ModuleList(
            [
                BaseTransformerLayer(d_model, n_heads, dim_feed_forward, dropout)
                for _ in range(num_encoder_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(d_model, n_heads, dim_feed_forward, dropout)
                for _ in range(num_decoder_layers)
            ]
        )
        self.linear_layer = nn.Linear(d_model, vocab_size)

    def encode(self, src_seq, src_mask):
        src_emb = self.embedding_layer(src_seq)
        src_emb = self.positional_encoding(src_emb)
        encoder_output = src_emb
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, src_mask)
        return encoder_output

    def decode(self, tgt_seq, encoder_output, src_mask, tgt_mask):
        tgt_emb = self.embedding_layer(tgt_seq)
        tgt_emb = self.positional_encoding(tgt_emb)
        decoder_output = tgt_emb
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output, src_mask, tgt_mask)
        return self.linear_layer(decoder_output)

    def forward(
        self,
        input,
        output,
        encoder_attention_mask=None,
        decoder_attention_mask=None,
    ):
        encoder_output = self.encode(input, encoder_attention_mask)
        return self.decode(
            output, encoder_output, encoder_attention_mask, decoder_attention_mask
        )
