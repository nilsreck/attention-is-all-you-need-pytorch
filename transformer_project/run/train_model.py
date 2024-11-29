import os
import sys

# Add the parent directory to the system path for importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformer_project.modelling.transformer import Transformer

model = Transformer(
    vocab_size=50000,
    d_model=512,
    n_heads=8,
    num_decoder_layers=4,
    num_encoder_layers=4,
    dim_feedforward=2048,
)

print(model.parameters)

bias_params = [p for name, p in model.named_parameters() if "bias" in name]
others = [p for name, p in model.named_parameters() if "bias" not in name]

optim.SGD(
    [{"params": others}, {"params": bias_params, "weight_decay": 0}],
    weight_decay=1e-2,
    lr=1e-2,
)
