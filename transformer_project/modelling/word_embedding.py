import torch.nn as nn

vocab_size = 50000
embedding_dim = 300
padding_idx = 0
embedding_layer = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
