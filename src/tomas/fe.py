# Implements GNN feature extractor for Sergi Tomás Martínez, 2025.
from fgproee.alg.fgp import buildLookaheadWeights
from sampler.randomcircuit import RandomCircuit
import torch
import torch.nn as nn
import copy
import gymnasium as gym
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class GCNN(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim=16):
        super(GCNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x # [B * N, hidden_dim]
    
    
class GAT(torch.nn.Module):
    """GATv2Conv-based Graph Attention Network. 
    For an explanation of GATv2Conv, see: https://nn.labml.ai/graphs/gatv2/index.html"""
    def __init__(self, input_dim: int, hidden_dim: int = 16, heads: int = 4):
        super(GAT, self).__init__()
        assert 16 % heads == 0
        per_head = hidden_dim // heads

        self.gat1 = GATv2Conv(input_dim, per_head, heads=heads, concat=True)
        self.gat2 = GATv2Conv(hidden_dim, hidden_dim, heads=1, concat=True)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        return x # [B * N, hidden_dim]


class FeatureExtractor(BaseFeaturesExtractor):
    """Feature Extractor for Graph Neural Networks.

    This class is implemented using a GNN, a pooling function, and concatenation with the additional components of the state.
    The pooling method is a simple readout function: the final hidden states of the nodes are concatenated from 0 to |Q|.
    The additional components of the state (remaining core capacities) are concatenated to the output of the GNN.
    It can either be a CGNN or a GAT.
    Here, by hidden dim we mean the target embedding size per qubit, which is 16

    Args:
        observation_space (spaces.Dict): The observation space of the environment.
        gnn (str): The type of GNN to use ('gcn' or 'gat').
        hidden_dim (int): The dimension of the hidden layers in the GNN.
        heads (int): The number of attention heads for GAT. Default is 4.
    """

    def __init__(self, observation_space: spaces.Dict, gnn="gcn", hidden_dim=16, heads=4):
        nf_shape = observation_space["node_features"].shape
        self.N_max = nf_shape[0]
        self.F_nf = nf_shape[1]
        self.C = observation_space["Z"].shape[0]
        self.gnn_type = gnn.lower()

        features_dim = self.N_max * hidden_dim + self.C  

        super().__init__(observation_space, features_dim)

        if self.gnn_type == 'gcn':
            self.gnn = GCNN(self.F_nf, hidden_dim=hidden_dim)
        elif self.gnn_type == 'gat':
            self.gnn = GAT(self.F_nf, hidden_dim=hidden_dim, heads=heads)
        else:
            raise ValueError("Unsupported GNN type. Use 'gcn' or 'gat'.")
        
        # Cache
        eye = torch.eye(self.N_max)
        self.register_buffer("_eye", eye, persistent=False)
        
    def _edge_index(self, A1: torch.Tensor, A2: torch.tensor) -> torch.Tensor:
        """Extracts the edge indexes from the A1 matrix, which has
        shape [B, N, N]. Possible future memory problems if A1 and A2 are dense. 
        E_total is the number of edges in the graph.
        NOTE: This could be achieved more directly with geometric operations
        like 'torch_geometric.utils.dense_to_sparse', but then we would need to
        process each batch element separately."""

        # It's not clear what to do with A1 and A2.
        # There are two proposals:
        # 1. Sum A1 and A2, and then extract the edge index.
        # 2. Concatenate A1 and A2, and then extract the edge index.
        # For now, I'm summing them.
        # GATv2Conv layers accept more than one edge index, so we can use both.
        # But not in GCNConv.

        B, N, _ = A1.shape
        device = A1.device

        A1 = A1.clone()
        diag = self._eye[:N, :N].unsqueeze(0)
        A1 = A1 * (1.0 - diag)
        A2 = A2 * (1.0 - diag)

        mask = (A1 > 0) | (A2 > 0)# [B,N,N]
        if not mask.any():
            edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
            return edge_index, None
        
        # Find non-zero entries in the across the batch
        b_idx, i_idx, j_idx = torch.nonzero(mask, as_tuple=True) # all edges across batch
        
        offset = (b_idx.to(torch.long) * N)
        src = offset + i_idx.to(torch.long) # Source nodes
        dst = offset + j_idx.to(torch.long) # Destination nodes

        edge_index = torch.stack([src, dst], dim=0)  # [2, E_total] edges of the disjoint graph

        # This is just to make undirected by adding the reverse edges
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1) # [2, 2E_total]

        if self.gnn_type == 'gcn':
            w = (A1[b_idx, i_idx, j_idx] + A2[b_idx, i_idx, j_idx])  # [E_total]
            edge_attr = torch.cat([w, w], dim=0)  # [2E_total]
            return edge_index, edge_attr
        
        elif self.gnn_type == 'gat':
            # For GAT, we need to return edge attributes as well
            A1 = A1[b_idx, i_idx, j_idx]
            A2 = A2[b_idx, i_idx, j_idx]
            edge_attr = torch.stack([A1, A2], dim=1) # [E_total, 2]
            # Add reverse edges
            edge_attr = torch.cat([edge_attr, edge_attr], dim=0)  # [2E_total, 2]
            return edge_index, edge_attr

        else:
            return edge_index, None


    def forward(self, x) -> torch.Tensor:
        device = next(self.parameters()).device

        #TODO: Not mentioned in the paper, but perhaps we should normalize A2
        #so that it has the same scale as A1. Gates in the current slice in A2 have
        #infinite weight. This isn't addressed in the paper, but doesn't make much 
        #sense outside of FGP-roEE.

        A1 = x["A1"].to(device)
        A2 = x["A2"].to(device)
        Z = x["Z"].to(device)
        X = x["node_features"].to(device)
        B, N, F_nf = X.shape

        edge_index, edge_attr = self._edge_index(A1, A2)
        X_flat = X.view(B * N, F_nf)

        if self.gnn_type == 'gat':
            edge_attr = edge_attr.to(device)
            gnn_output_flat = self.gnn(X_flat, edge_index, edge_attr=edge_attr)
        elif self.gnn_type == 'gcn':
            edge_attr = edge_attr.to(device)
            gnn_output_flat = self.gnn(X_flat, edge_index, edge_weight=edge_attr)
        else:
            gnn_output_flat = self.gnn(X_flat, edge_index)

        gnn_output = gnn_output_flat.view(B, -1)
        combined_features = torch.cat((gnn_output, Z), dim=1) # [B, N * hidden_dim + C]
        return combined_features
    

