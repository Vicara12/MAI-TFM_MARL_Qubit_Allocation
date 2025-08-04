from typing import Tuple, List
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
from utils.customtypes import Hardware, Circuit
from utils.circuitutils import getCircuitMatrices2xE



class GNNEncoder(torch.nn.Module):
  ''' Handles the codification of circuit slices via GNN.
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


  def encodeCircuits(self, circuits: List[Circuit]):
    slices_per_circuit = [c.n_slices for c in circuits]
    indices = [sum(slices_per_circuit[:i]) for i in range(len(circuits))]
    # Each circuit is composed as a list of matrices, join all lists into a single mega-list
    matrices = [m for c in circuits for m in getCircuitMatrices2xE(c)]
    result = self.forward(matrices)
    # Split the resulting tensor into a list with one tensor per circuit
    slice_embs = [result[i_ini:i_fi,:] for i_ini, i_fi in zip(indices[:-1], indices[1:])]
    circuit_embs = []
    # Each item in slice embs is a tensor. The ith row of this tensor corresponds to the embedding
    # of the ith time slice. We want a tensor in which the ith row contains a circuit embedding from
    # the ith time slice until the end (through pooling)
    for circuit in slice_embs:
      circuit_emb = torch.empty_like(circuit)
      for i in range(circuit.shape[0]):
        circuit_emb[i,:] = torch.max(circuit[i:,:], dim=0).values
      circuit_embs.append(circuit_emb)
    return slice_embs