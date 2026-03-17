# DeepSets-style Siamese encoder with self-attention over eigenvalues only.

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSetSelfAttentionEigenOnly(nn.Module):
    """
    Siamese DeepSets encoder that uses only eigenvalues.
    Inputs:  eigenvalues (B, S, input_dim)
    Outputs: embeddings (B, output_dim)
    """

    def __init__(self, input_dim=1, hidden_dim=64, output_dim=64, phi_layers=20, rho_layers=12):
        super(DeepSetSelfAttentionEigenOnly, self).__init__()

        # Phi: elementwise MLP applied to each eigenvalue.
        def build_phi(in_dim, out_dim, layers):
            mlp = []
            current_in = in_dim
            for _ in range(layers):
                mlp.append(nn.Linear(current_in, out_dim))
                mlp.append(nn.ReLU())
                current_in = out_dim
            return nn.Sequential(*mlp)

        self.phi_eigenvalues = build_phi(input_dim, hidden_dim, phi_layers)

        # Self-attention projections (Q, K, V).
        self.query_eigen = nn.Linear(hidden_dim, hidden_dim)
        self.key_eigen = nn.Linear(hidden_dim, hidden_dim)
        self.value_eigen = nn.Linear(hidden_dim, hidden_dim)

        # Rho: aggregation MLP after attention pooling.
        def build_rho(in_dim, out_dim, layers):
            mlp = []
            current_in = in_dim
            for _ in range(layers - 1):
                mlp.append(nn.Linear(current_in, hidden_dim))
                mlp.append(nn.ReLU())
                current_in = hidden_dim
            mlp.append(nn.Linear(current_in, out_dim))
            return nn.Sequential(*mlp)

        self.rho = build_rho(hidden_dim, output_dim, rho_layers)

    def elementwise_phi(self, x):
        """
        Apply phi to each element.
        Input:  (B, S, input_dim)
        Output: (B, S, hidden_dim)
        """
        BS, SS, ID = x.shape
        x = x.view(BS * SS, ID)
        x = self.phi_eigenvalues(x)
        x = x.view(BS, SS, -1)
        return x

    def self_attention_block(self, x):
        """
        Self-attention pooling over the sequence.
        Input:  (B, S, hidden_dim)
        Output: (B, hidden_dim)
        """
        BS, SS, HD = x.shape
        Q = self.query_eigen(x.view(BS * SS, HD)).view(BS, SS, HD)
        K = self.key_eigen(x.view(BS * SS, HD)).view(BS, SS, HD)
        V = self.value_eigen(x.view(BS * SS, HD)).view(BS, SS, HD)

        scores = torch.matmul(Q, K.transpose(1, 2)) / (HD ** 0.5)
        weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(weights, V).sum(dim=1)
        return attended

    def forward_once(self, eigenvalues):
        """
        Encode a single input.
        """
        eig_features = self.elementwise_phi(eigenvalues)
        attended_eig = self.self_attention_block(eig_features)
        output = self.rho(attended_eig)
        return output

    def forward(self, eigenvalues1, eigenvalues2):
        """
        Siamese forward: returns two embeddings.
        """
        out1 = self.forward_once(eigenvalues1)
        out2 = self.forward_once(eigenvalues2)
        return out1, out2
