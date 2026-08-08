import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import CasualtiesEnv
import Encoders
import ObservationNormalization as ObsNorm
import Types


class CasualtiesFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        features_dim = (
            Encoders.GENERAL_EMBEDDING_DIM + Encoders.GRID_EMBEDDING_DIM
        )
        super().__init__(observation_space, features_dim=features_dim)

        self.general_encoder = Encoders.GeneralEncoder(
            CasualtiesEnv._general_input_dim,
            Encoders.GENERAL_EMBEDDING_DIM,
        )
        self.spatial_encoder = Encoders.SpatialEncoder(
            in_channels=ObsNorm.SPATIAL_CHANNELS,
            width=Types.SIGHT_RANGE_X * 2 + 1,
            height=Types.SIGHT_RANGE_Y * 2 + 1,
            embedding_dim=Encoders.GRID_EMBEDDING_DIM,
        )

    def forward(self, observations):
        general = self.general_encoder(observations["general"])
        spatial = self.spatial_encoder(observations["spatial"])
        return torch.cat([spatial, general], dim=-1)
