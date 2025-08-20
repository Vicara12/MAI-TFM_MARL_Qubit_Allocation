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

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x # [N, hidden_dim]
    
    
class GAT(torch.nn.Module):
    """GATv2Conv-based Graph Attention Network. 
    For an explanation of GATv2Conv, see: https://nn.labml.ai/graphs/gatv2/index.html"""
    def __init__(self, input_dim: int, hidden_dim: int = 16, heads: int = 4):
        super(GAT, self).__init__()
        assert 16 % heads == 0
        per_head = hidden_dim // heads

        self.gat1 = GATv2Conv(input_dim, per_head, heads=heads, concat=True)
        self.gat2 = GATv2Conv(hidden_dim, hidden_dim, heads=1, concat=True)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = self.gat2(x, edge_index)
        return x # [N, hidden_dim]


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

        features_dim = self.F_nf * hidden_dim + self.C  # Node features + core capacities

        super().__init__(observation_space, features_dim)

        if gnn.lower() == 'gcn':
            self.gnn = GCNN(self.F_nf, hidden_dim=hidden_dim)
        elif gnn.lower() == 'gat':
            self.gnn = GAT(self.F_nf, hidden_dim=hidden_dim, heads=heads)
        else:
            raise ValueError("Unsupported GNN type. Use 'gcn' or 'gat'.")
        

    def edge_index(self, A1_slice: torch.Tensor) -> torch.Tensor:
        """Extracts the edge index from the A1 matrix."""
        A = A1_slice["edge_index"]
        # Remove self-loop edges
        A.fill_diagonal_(0)
        I, J = torch.nonzero(A > 0.0, as_tuple=True)

        return torch.stack([I, J], dim=0).long()


    def forward(self, x) -> torch.Tensor:
        device = next(self.parameters()).device

        A1 = x["A1"].to(device)
        Z = x["Z"].to(device)
        X = x["node_features"].to(device)
        N = x["N"].to(device).view(-1).long()

        edge_index = self.edge_index(x)
        gnn_output = self.gnn(x, edge_index)
        combined_features = torch.cat((gnn_output, Z), dim=-1)
        return self.fc(combined_features)
    

