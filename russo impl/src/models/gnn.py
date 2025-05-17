from typing import Tuple
import torch
import torch.nn as nn



class GNN(nn.Module):
  ''' Graph Neural Network based on reference [1].

  References:
    [Semi-Supervised Classification with Graph Convolutional Networks]
    (https://arxiv.org/abs/1609.02907).
      Thomas N. Kipf, Max Welling. 2016.
  '''
  @classmethod
  def _graphApproxLagrangian(graph: torch.ndarray) -> torch.ndarray:
    assert len(graph.shape) == 2, "Graph matrix should have two dims"
    assert len(set(graph.shape)) == 1, "Graph matrix should be square"
    Z_ = graph + torch.eye(graph.shape[0]) # Add self loops
    D_sqrt = torch.diag(torch.pow(torch.sum(Z_, axis=-1), -0.5)) # Compute D^(-0.5)
    return D_sqrt @ Z_ @ D_sqrt

  def __init__(self, graphs: Tuple[torch.ndarray], out_shape: int):
    super().__init__()
    self.deg = graphs[0].shape[0]
    self.T = len(graphs)
    self.out_shape = out_shape
    self.fc = nn.Linear(self.deg, self.out_shape)
    # Register graph matrices as a 3D tensor of shape [T,Q,Q]
    self.laplacians = torch.stack(
      torch.from_numpy(GNN._graphApproxLagrangian(graph)).float() for graph in graphs
    )
    self.register_buffer("graphs_tensor", self.laplacians)
  
  def forward(self, X):
    """
    X shape: [Q, E] where Q = num logical qubits, E = qubit embedding dimension
    Returns [T, Q, O] where T = num slices, Q = num logical qubits, O = output embedding dimension
    """
    # This basically performs L_t @ X @ W for all t in batch
    X = X.unsqueeze(0).expand(self.T, -1, -1) # shape [T, Q, d_in]
    H = torch.bmm(self.laplacians, X)    # [T, Q, d_in]
    return self.fc(H)                    # [T, Q, d_out]