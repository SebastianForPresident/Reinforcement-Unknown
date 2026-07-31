import torch
import torch.nn as nn
import numpy as np

GENERAL_EMBEDDING_DIM = 64

GRID_CHANNELS = 32
GRID_EMBEDDING_DIM = 64

class GeneralEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU()
        )

    def forward(self, input):
        return self.net(input)

class GridEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),

            nn.Linear(out_channels, embedding_dim),
            nn.ReLU()
        )

    def forward(self, input):
        return self.net(input)

