import torch
from torch.nn import Linear


def test_linear_layer():
    # Set seed for reproducibility
    seed = 42
    torch.manual_seed(seed)

    # Define linear layer
    layer = Linear(2, 3)

    # Generate random weight and bias vectors for the linear layer
    weight = torch.randn(layer.weight.size())
    bias = torch.randn(layer.bias.size())
    layer.weight = torch.nn.Parameter(weight)
    layer.bias = torch.nn.Parameter(bias)

    x = torch.randn(5, 2)

    # Compute the expected and actual outputs
    expected = torch.matmul(weight, x.T).T + bias.unsqueeze(0).repeat((x.size(0), 1))
    actual = layer(x)

    assert torch.allclose(expected, actual)
