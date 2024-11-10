import torch.nn as nn


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()

        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ffn, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
