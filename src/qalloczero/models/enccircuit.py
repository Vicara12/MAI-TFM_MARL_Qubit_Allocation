from typing import Tuple, List
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
from utils.customtypes import Hardware


class CircuitEncoder(torch.nn.Module):
  ''' Handles the codification of circuits as slices of circuit embeddings.
  '''

  def __init__(self, hardware: Hardware, nn_dims: Tuple[int]):
    assert len(nn_dims) > 2, "At least input and output dimensions must be specified"
    assert all(d > 0 for d in nn_dims), "All nn_dims must be strictly positive"
    super().__init__()
    self.qubit_embeddings = torch.nn.Parameter(torch.randn(hardware.n_physical_qubits+1, nn_dims[0]),
                                               requires_grad=True)
    self.convs = torch.nn.ModuleList()
    for in_dim, out_dim in zip(nn_dims[:-1], nn_dims[1:]):
      self.convs.append(GCNConv(in_dim, out_dim))


  def forward(self, slice_matrices: List[torch.Tensor]) -> torch.Tensor:
    batch = Batch.from_data_list(
      [Data(x=self.qubit_embeddings, edge_index=mat) for mat in slice_matrices])
    x = batch.x
    for conv in self.convs:
      x = conv(x, batch.edge_index)
      x = torch.relu(x)
    return x