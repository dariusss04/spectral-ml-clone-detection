# DeepSets-based Siamese encoder on eigenvalues only.

import torch
import torch.nn as nn


class DeepSetPhi(nn.Module):
    """
    Elementwise DeepSets phi network.
    Inputs:  x (B, S, input_dim)
    Outputs: (B, S, hidden_dim)
    """

    def __init__(self, input_dim, hidden_dim, num_layers=5):
        super(DeepSetPhi, self).__init__()
        layers = []
        in_features = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU())
            in_features = hidden_dim
        self.phi = nn.Sequential(*layers)

    def forward(self, x):
        """
        Apply phi to each set element.
        Input:  (B, S, input_dim)
        Output: (B, S, hidden_dim)
        """
        BS, SS, ID = x.shape
        x = x.view(BS * SS, ID)
        x = self.phi(x)
        x = x.view(BS, SS, -1)
        return x


class SiameseNetworkDeepSetsEigenOnly(nn.Module):
    """
    Siamese DeepSets encoder using only eigenvalues.
    Inputs:  eigenvalues (B, S, 1)
    Outputs: embeddings (B, output_dim)
    """

    def __init__(self, input_dim=1, hidden_dim=64, output_dim=64, phi_layers=5, rho_layers=5):
        super(SiameseNetworkDeepSetsEigenOnly, self).__init__()

        # Phi: elementwise transformation for eigenvalues.
        self.phi_eigenvalues = DeepSetPhi(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=phi_layers)

        # Rho: aggregation MLP after summation.
        rho_layers_list = []
        in_features = hidden_dim
        layer_dims = [256, 128, 128, 128, 64]

        for dim in layer_dims:
            rho_layers_list.append(nn.Linear(in_features, dim))
            rho_layers_list.append(nn.ReLU())
            in_features = dim

        rho_layers_list.append(nn.Linear(in_features, output_dim))
        self.rho = nn.Sequential(*rho_layers_list)

    def forward_once(self, eigenvalues):
        """
        Encode a single input.
        """
        phi_eigen = self.phi_eigenvalues(eigenvalues)
        aggregated_eigen = phi_eigen.sum(dim=1)
        output = self.rho(aggregated_eigen)
        return output

    def forward(self, eigenvalues1, eigenvalues2):
        """
        Siamese forward: returns two embeddings.
        """
        output1 = self.forward_once(eigenvalues1)
        output2 = self.forward_once(eigenvalues2)
        return output1, output2
