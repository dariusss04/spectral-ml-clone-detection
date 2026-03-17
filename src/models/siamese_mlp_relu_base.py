# Base Siamese MLP with configurable depth using ReLU activations.

import torch
import torch.nn as nn


class BaseSiameseNetwork(nn.Module):
    """
    Configurable-depth Siamese MLP with ReLU.
    Inputs:  x (B, input_size)
    Outputs: embeddings (B, 64)
    """

    def __init__(self, num_layers=2, input_size=200):
        super(BaseSiameseNetwork, self).__init__()

        layers = [nn.Flatten(), nn.Linear(input_size, 1024), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(1024, 1024))
            layers.append(nn.ReLU())

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


