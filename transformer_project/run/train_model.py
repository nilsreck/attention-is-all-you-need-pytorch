import torch.optim as optim
from transformer_project.modelling.transformer import Transformer

model = Transformer(
    vocab_size=50000,
    d_model=32,
    n_heads=8,
    num_decoder_layers=2,
    num_encoder_layers=2,
    dim_feedforward=128,
    dropout=0.0,
    maxlen=8,
)

params_with_decay = []
params_without_decay = []

no_decay = ["bias", "layer_norm"]
for name, param in model.named_parameters():
    if any(nd in name for nd in no_decay):
        params_without_decay.append(param)
    else:
        params_with_decay.append(param)

optimizer = optim.AdamW(
    [
        {"params": params_with_decay},
        {"params": params_without_decay, "weight_decay": 0.0},
    ],
    lr=1e-4,
    weight_decay=1e-2,
)
