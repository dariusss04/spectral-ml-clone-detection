# 9-layer Siamese MLP with LeakyReLU + LayerNorm.

import torch
import torch.nn as nn


class SiameseNetwork_LeakyReLU_LayerNorm(nn.Module):
    """
    Siamese MLP with LeakyReLU + LayerNorm.
    Inputs:  x (B, 200)
    Outputs: embeddings (B, 64)
    """

    def __init__(self):
        super(SiameseNetwork_LeakyReLU_LayerNorm, self).__init__()
        layers = []

        # Input + hidden stack.
        layers.append(nn.Flatten())
        layers.append(nn.Linear(200, 1024))
        layers.append(nn.LeakyReLU())
        layers.append(nn.LayerNorm(1024))

        for _ in range(7):
            layers.append(nn.Linear(1024, 1024))
            layers.append(nn.LeakyReLU())
            layers.append(nn.LayerNorm(1024))

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
        output1 = self.forward_once(x1)
        output2 = self.forward_once(x2)
        return output1, output2

