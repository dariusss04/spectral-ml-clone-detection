# 9-layer Siamese MLP with ReLU + BatchNorm.

import torch
import torch.nn as nn


class SiameseNetwork_ReLU_BatchNorm_9L(nn.Module):
    """
    Siamese MLP with ReLU + BatchNorm.
    Inputs:  x (B, 200)
    Outputs: embeddings (B, 64)
    """

    def __init__(self):
        super(SiameseNetwork_ReLU_BatchNorm_9L, self).__init__()
        layers = []

        # Input + hidden stack.
        layers.append(nn.Flatten())
        layers.append(nn.Linear(200, 1024))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(1024))

        for _ in range(7):
            layers.append(nn.Linear(1024, 1024))
            layers.append(nn.ReLU())
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



