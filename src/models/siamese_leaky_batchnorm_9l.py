# 9-layer Siamese MLP with LeakyReLU + BatchNorm.

import torch
import torch.nn as nn


class SiameseNetworkBatchNorm9L(nn.Module):
    """
    Siamese MLP with LeakyReLU + BatchNorm.
    Inputs:  x (B, input_size)
    Outputs: embeddings (B, 64)
    """

    def __init__(self, input_size=200):
        super(SiameseNetworkBatchNorm9L, self).__init__()
        layers = []

        # Input + hidden stack.
        layers.append(nn.Flatten())
        layers.append(nn.Linear(input_size, 1024))
        layers.append(nn.LeakyReLU(negative_slope=0.01))
        layers.append(nn.BatchNorm1d(1024))

        for _ in range(7):
            layers.append(nn.Linear(1024, 1024))
            layers.append(nn.LeakyReLU(negative_slope=0.01))
            layers.append(nn.BatchNorm1d(1024))

        # Output projection.
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
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        return out1, out2


