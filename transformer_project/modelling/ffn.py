import torch.nn as nn


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ffn), nn.ReLU(), nn.Linear(d_ffn, d_model)
        )

        def forward(self, x):
            return self.feed_forward(x) 
