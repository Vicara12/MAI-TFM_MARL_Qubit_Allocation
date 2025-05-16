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

  def __init__(self, graphs: Tuple[torch.ndarray], out_shape):
    super().__init__()
    self.deg = graphs[0].shape[0]
    self.out_shape = out_shape
    self.fc = nn.Linear(self.deg, self.out_shape)
    # Register graph matrices
    self.laplacians_ = tuple(
      torch.from_numpy(GNN._graphApproxLagrangian(graph)).float() for graph in graphs
    )
    for t, graph in enumerate(self.laplacians_):
      self.register_buffer(f"graph_matrix_{t}", graph)
  
  def forward(self, X, t):
    y = torch.mm(self.laplacians_[t], X)
    return self.fc(y)