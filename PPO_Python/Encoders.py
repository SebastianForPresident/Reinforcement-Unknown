"""Full-resolution CB1 feature encoders."""

import torch.nn as nn


GENERAL_HIDDEN_DIM = 256
GENERAL_EMBEDDING_DIM = 128
GRID_CHANNELS = 32
GRID_EMBEDDING_DIM = 128


class GeneralEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=GENERAL_EMBEDDING_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, GENERAL_HIDDEN_DIM),
            nn.LayerNorm(GENERAL_HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(GENERAL_HIDDEN_DIM, GENERAL_HIDDEN_DIM),
            nn.LayerNorm(GENERAL_HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(GENERAL_HIDDEN_DIM, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
        )

    def forward(self, inputs):
        return self.net(inputs)


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(8, channels),
        )
        self.activation = nn.SiLU()

    def forward(self, inputs):
        return self.activation(inputs + self.net(inputs))


class SpatialEncoder(nn.Module):
    """Joint terrain/fluid encoder that never pools or changes 85x49 geometry."""

    def __init__(self, in_channels, width, height, embedding_dim=GRID_EMBEDDING_DIM):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                GRID_CHANNELS,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(8, GRID_CHANNELS),
            nn.SiLU(),
        )
        self.residual = nn.Sequential(
            DilatedResidualBlock(GRID_CHANNELS, dilation=1),
            DilatedResidualBlock(GRID_CHANNELS, dilation=2),
            DilatedResidualBlock(GRID_CHANNELS, dilation=4),
            DilatedResidualBlock(GRID_CHANNELS, dilation=8),
            DilatedResidualBlock(GRID_CHANNELS, dilation=16),
        )
        self.project = nn.Sequential(
            nn.Flatten(),
            nn.Linear(GRID_CHANNELS * width * height, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
        )

    def forward(self, inputs):
        features = self.stem(inputs)
        features = self.residual(features)
        return self.project(features)
