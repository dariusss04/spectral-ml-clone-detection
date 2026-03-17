# 12-layer Siamese MLP with LeakyReLU + LayerNorm.

import torch
import torch.nn as nn


class SiameseNetworkLayerNorm(nn.Module):
    """
    Siamese MLP with LeakyReLU + LayerNorm.
    Inputs:  x (B, input_size)
    Outputs: embeddings (B, 64)
    """

    def __init__(self, input_size=200):
        super(SiameseNetworkLayerNorm, self).__init__()
        layers = [
            nn.Flatten(),
            nn.Linear(input_size, 1024),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(1024),
        ]
        for _ in range(10):
            layers.extend([
                nn.Linear(1024, 1024),
                nn.LeakyReLU(negative_slope=0.01),
                nn.LayerNorm(1024),
            ])
        layers.append(nn.Linear(1024, 64))

        self.fc = nn.Sequential(*layers)

    def forward_once(self, x):
        """
        Encode a single input.
        """
        return self.fc(x)

    def forward(self, x1, x2):
        """
        Siamese forward: returns two embeddings.
        """
        output1 = self.forward_once(x1)
        output2 = self.forward_once(x2)
        return output1, output2

