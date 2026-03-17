# DeepSets-style Siamese encoder with self-attention over eigenvalues and edge counts.

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSetSelfAttention(nn.Module):
    """
    Siamese DeepSets encoder with self-attention for eigenvalues and edge counts.
    Inputs:  eigenvalues/num_edges (B, S, input_dim)
    Outputs: embeddings (B, output_dim)
    """

    def __init__(self, input_dim=1, hidden_dim=64, output_dim=64, phi_layers=4, rho_layers=4):
        super(DeepSetSelfAttention, self).__init__()

        # Phi: elementwise MLP for each set element.
        def build_phi(in_dim, out_dim, layers):
            mlp = []
            current_in = in_dim
            for _ in range(layers):
                mlp.append(nn.Linear(current_in, out_dim))
                mlp.append(nn.ReLU())
                current_in = out_dim
            return nn.Sequential(*mlp)

        self.phi_eigenvalues = build_phi(input_dim, hidden_dim, phi_layers)
        self.phi_num_edges = build_phi(input_dim, hidden_dim, phi_layers)

        # Self-attention projections for eigenvalues and edge counts.
        self.query_eigenvalues = nn.Linear(hidden_dim, hidden_dim)
        self.key_eigenvalues = nn.Linear(hidden_dim, hidden_dim)
        self.value_eigenvalues = nn.Linear(hidden_dim, hidden_dim)

        self.query_num_edges = nn.Linear(hidden_dim, hidden_dim)
        self.key_num_edges = nn.Linear(hidden_dim, hidden_dim)
        self.value_num_edges = nn.Linear(hidden_dim, hidden_dim)

        # Rho: aggregation MLP after concatenation.
        def build_rho(in_dim, out_dim, layers):
            mlp = []
            current_in = in_dim
            mlp.append(nn.Linear(current_in, hidden_dim))
            mlp.append(nn.ReLU())
            current_in = hidden_dim
            mlp.append(nn.Linear(current_in, out_dim))
            return nn.Sequential(*mlp)

        self.rho = build_rho(hidden_dim * 2, output_dim, rho_layers)

    def elementwise_phi(self, x, phi_network):
        """
        Apply phi to each element.
        Input:  (B, S, input_dim)
        Output: (B, S, hidden_dim)
        """
        BS, SS, ID = x.shape
        x = x.view(BS * SS, ID)
        x = phi_network(x)
        x = x.view(BS, SS, -1)
        return x

    def self_attention_block(self, x, query_proj, key_proj, value_proj):
        """
        Self-attention pooling over the sequence.
        Input:  (B, S, hidden_dim)
        Output: (B, hidden_dim)
        """
        BS, SS, HD = x.shape
        Q = query_proj(x.view(BS * SS, HD)).view(BS, SS, HD)
        K = key_proj(x.view(BS * SS, HD)).view(BS, SS, HD)
        V = value_proj(x.view(BS * SS, HD)).view(BS, SS, HD)

        scores = torch.matmul(Q, K.transpose(1, 2))
        weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(weights, V)
        aggregated = attended.sum(dim=1)
        return aggregated

    def forward_once(self, eigenvalues, num_edges):
        """
        Encode a single input (eigenvalues + edge counts).
        """
        eig = self.elementwise_phi(eigenvalues, self.phi_eigenvalues)
        attended_eig = self.self_attention_block(
            eig, self.query_eigenvalues, self.key_eigenvalues, self.value_eigenvalues
        )

        edges = self.elementwise_phi(num_edges, self.phi_num_edges)
        attended_edges = self.self_attention_block(
            edges, self.query_num_edges, self.key_num_edges, self.value_num_edges
        )

        concatenated = torch.cat([attended_eig, attended_edges], dim=-1)
        output = self.rho(concatenated)
        return output

    def forward(self, eigenvalues1, num_edges1, eigenvalues2, num_edges2):
        """
        Siamese forward: returns two embeddings.
        """
        out1 = self.forward_once(eigenvalues1, num_edges1)
        out2 = self.forward_once(eigenvalues2, num_edges2)
        return out1, out2
