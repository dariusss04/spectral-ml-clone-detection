# 9-layer Siamese MLP with Tanh + LayerNorm + Dropout.

import torch
import torch.nn as nn


class SiameseNetwork_Tanh_LayerNorm_Dropout_9L(nn.Module):
    """
    Siamese MLP with Tanh + LayerNorm + Dropout.
    Inputs:  x (B, input_size)
    Outputs: embeddings (B, 64)
    """

    def __init__(self, input_size=200, dropout_prob=0.3):
        super(SiameseNetwork_Tanh_LayerNorm_Dropout_9L, self).__init__()
        layers = []

        # Input + hidden stack.
        layers.append(nn.Flatten())
        layers.append(nn.Linear(input_size, 1024))
        layers.append(nn.Tanh())
        layers.append(nn.LayerNorm(1024))
        layers.append(nn.Dropout(p=dropout_prob))

        for _ in range(7):
            layers.append(nn.Linear(1024, 1024))
            layers.append(nn.Tanh())
            layers.append(nn.LayerNorm(1024))
            layers.append(nn.Dropout(p=dropout_prob))

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
