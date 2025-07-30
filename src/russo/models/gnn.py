from typing import Tuple
import torch
import torch.nn as nn
import copy



class GNN(nn.Module):
  ''' Graph Neural Network based on reference [1].

  This GNN implementation is capable of executing the same input with several graphs in batch. This
  is the case when running the circuit encoder with the qubits embeddings as inputs: feeding all
  circuit slices in a list means that when calling forward the GNN is run with all circuit slices in
  batch.

  Args:
    - emb_shape: shape of the embeddings.

  References:
    [Semi-Supervised Classification with Graph Convolutional Networks]
    (https://arxiv.org/abs/1609.02907).
      Thomas N. Kipf, Max Welling. 2016.
  '''

  def __init__(self, emb_shape: int):
    super().__init__()
    self.emb_shape = emb_shape
    self.fc = nn.Linear(self.emb_shape, self.emb_shape)
    
  
  def setGraphs(self, graphs: torch.Tensor):
    ''' Set the graphs used for the GNN as a 3D tensor of shape [T,Q,Q] or [B,T,Q,Q]. No self-loops.
    '''
    assert (graphs.diagonal(dim1=-2,dim2=-1) == 0).all(), "diagonal of graphs should be zeros (no self-loops)"
    assert len(graphs.shape) in (3,4), "graphs should be a [B,T,Q,Q] tensor (qubit graphs) or a [B,N,N] tensor (qubit cores)"
    assert (graphs.shape[-2] == graphs.shape[-1]), "last two dims of shape should match"
    self.B = graphs.shape[0]
    self.T = graphs.shape[1] if len(graphs.shape) == 4 else self.B
    self.N = graphs.shape[-1]
    dev = graphs.device
    # Handle buffer registration and prevent repeated attribute error
    if "laplacians" in self._buffers:
      if self.laplacians.shape == graphs.shape:
        self.laplacians = copy.deepcopy(graphs)
      else:
        del self._buffers['laplacians']
        self.register_buffer("laplacians", copy.deepcopy(graphs))
    else:
      self.register_buffer("laplacians", copy.deepcopy(graphs))
    # Below I have a batch of adjacency graphs (that is, a 3D tensor), and I compute the Laplacian
    # of each adjacency graph in batch. If at any point it looks confusing just keep in mind that
    # I'm applying the same operation to all matrices in the batch
    repeats = [self.T, 1, 1] if len(graphs.shape) == 3 else [self.B, self. T, 1, 1]
    self.laplacians += torch.eye(self.N, device=dev).repeat(*repeats) # Add self loops to all graphs in batch
    D_sqrt = torch.diag_embed(torch.pow(torch.sum(self.laplacians, axis=-1),-0.5))
    self.laplacians = D_sqrt @ self.laplacians @ D_sqrt # D^{-0.5} * Z * D^{-0.5}
    

  def clearGraphs(self):
    del self._buffers['laplacians']
  
  
  def forward(self, X) -> torch.Tensor:
    '''
    X shape: [B, Q, E] where Q = num logical qubits, E = qubit embedding dimension
    Returns [B, T, Q, O] where T = num slices, Q = num logical qubits, O = output embedding dimension
    or alternatively [B, T, C, O] where C = number of cores and O = output embedding dimension.
    '''
    if 'laplacians' not in self._buffers:
      raise Exception("graphs have not been set, call setGraphs before forward")
    # This performs L_t @ X @ W
    X = X.unsqueeze(1).expand(-1, self.T, -1, -1) # shape [B, T, Q, d_in]
    H = self.laplacians @ X # [B, T, Q, d_in] - uses broadcasting 
    return self.fc(H) # [B, T, Q, d_out]