import torch
import numpy as np
import Encoders
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import CasualtiesEnv
import Types

class CasualtiesFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=(
            Encoders.GENERAL_EMBEDDING_DIM
            + 2 * Encoders.GRID_EMBEDDING_DIM
            )
        )

        self.general_encoder = Encoders.GeneralEncoder(CasualtiesEnv._general_input_dim, Encoders.GENERAL_EMBEDDING_DIM)

        self.block_grid_encoder = Encoders.GridEncoder(
            in_channels=len(Types.BLOCK_DTYPE.names),
            out_channels=Encoders.GRID_CHANNELS,
            embedding_dim=Encoders.GRID_EMBEDDING_DIM,
        )

        self.fluid_grid_encoder = Encoders.GridEncoder(
                in_channels=len(Types.FLUID_TILE_DTYPE.names),
                out_channels=Encoders.GRID_CHANNELS,
                embedding_dim=Encoders.GRID_EMBEDDING_DIM,
            )

    def forward(self, observations):
        general = observations["general"]
        blocks = observations["blocks"]
        fluids = observations["fluids"]

        general = self.general_encoder(general)
        blocks = self.block_grid_encoder(blocks)
        fluids = self.fluid_grid_encoder(fluids)

        return torch.cat([blocks, fluids, general], dim=-1)