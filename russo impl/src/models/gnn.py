from typing import Tuple
import torch
import torch.nn as nn



class GNN(nn.Module):
  ''' Graph Neural Network based on reference [1].

  This GNN implementation is capable of executing the same input with several graphs in batch. This
  is the case when running the circuit encoder with the qubits embeddings as inputs: feeding all
  circuit slices in a list means that when calling forward the GNN is run with all circuit slices in
  batch.

  Args:
    - deg: number of nodes in the graph.
    - out_shape: shape of the output.

  References:
    [Semi-Supervised Classification with Graph Convolutional Networks]
    (https://arxiv.org/abs/1609.02907).
      Thomas N. Kipf, Max Welling. 2016.
  '''
  @classmethod
  def _graphApproxLagrangian(graph: torch.Tensor) -> torch.Tensor:
    assert len(graph.shape) == 2, "Graph matrix should have two dims"
    assert len(set(graph.shape)) == 1, "Graph matrix should be square"
    Z_ = graph + torch.eye(graph.shape[0]) # Add self loops
    D_sqrt = torch.diag(torch.pow(torch.sum(Z_, axis=-1), -0.5)) # Compute D^(-0.5)
    return D_sqrt @ Z_ @ D_sqrt

  def __init__(self, deg: int, out_shape: int):
    super().__init__()
    self.deg = deg
    self.out_shape = out_shape
    self.fc = nn.Linear(self.deg, self.out_shape)
    self.laplacians = None
    
  
  def setGraphs(self, graphs: Tuple[torch.Tensor]):
    ''' Set the graphs used for the GNN as a 3D tensor of shape [T,Q,Q]. No self-loops.
    '''
    self.laplacians = torch.stack(
      torch.from_numpy(GNN._graphApproxLagrangian(graph)).float() for graph in graphs
    )
    self.T = len(graphs)


  def clearGraphs(self):
    self.laplacians = None
  
  
  def forward(self, X) -> torch.Tensor:
    '''
    X shape: [Q, E] where Q = num logical qubits, E = qubit embedding dimension
    Returns [T, Q, O] where T = num slices, Q = num logical qubits, O = output embedding dimension
    '''
    if self.laplacians is None:
      raise Exception("graphs have not been set, call setGraphs before forward")
    # This performs L_t @ X @ W for all t in batch
    X = X.unsqueeze(0).expand(self.T, -1, -1) # shape [T, Q, d_in]
    H = torch.bmm(self.laplacians, X)    # [T, Q, d_in]
    return self.fc(H)                    # [T, Q, d_out]