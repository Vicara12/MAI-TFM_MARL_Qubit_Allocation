from typing import Tuple, List
import torch
from math import log
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse
from utils.customtypes import Hardware, Circuit
from utils.circuitutils import getCircuitMatrices2xE



class GNNEncoder(torch.nn.Module):
  ''' Handles the codification of circuit slices via GNN.
  '''

  @staticmethod
  def getPositionalEmbedding(T, d_model):
    position = torch.arange(T).unsqueeze(1)  # [T, 1]
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-log(10000.0) / d_model))
    pe = torch.zeros(T, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # [T, d_model]
  

  def __init__(self, hardware: Hardware, nn_dims: Tuple[int], qubit_embs: torch.Tensor):
    assert len(nn_dims) > 2, "At least input and output dimensions must be specified"
    assert all(d > 0 for d in nn_dims), "All nn_dims must be strictly positive"
    assert len(qubit_embs.shape) == 2, "Qubit embeddings must be a vector of embeddings (rank 2)"
    assert qubit_embs.shape[0] == hardware.n_physical_qubits, \
      f"Num of embeddings ({qubit_embs.shape[0]}) does not match num of physical qubits {hardware.n_physical_qubits}"
    assert qubit_embs.shape[1] == nn_dims[0], \
      f"Embedding size ({qubit_embs.shape[1]}) does not match nn_dims.shape[0] ({nn_dims.shape[0]})"
    super().__init__()
    self.hw = hardware
    self.nn_dims = nn_dims
    self.qubit_embeddings = qubit_embs
    self.convs = torch.nn.ModuleList()
    for in_dim, out_dim in zip(nn_dims[:-1], nn_dims[1:]):
      self.convs.append(GCNConv(in_dim, out_dim))


  def forward(self, slice_matrices: List[torch.Tensor]) -> torch.Tensor:
    # TODO: check zeros
    batch = Batch.from_data_list(
      [Data(x=self.qubit_embeddings, edge_index=mat) for mat in slice_matrices])
    x = batch.x
    for conv in self.convs:
      x = conv(x, batch.edge_index)
      x = torch.relu(x)
    return x


  def encodeCircuits(self, circuits: List[Circuit]) -> Tuple[torch.Tensor, torch.Tensor]:
    slices_per_circuit = [c.n_slices for c in circuits]
    indices = [sum(slices_per_circuit[:i]) for i in range(len(circuits)+1)]
    # Each circuit is composed as a list of matrices, join all lists into a single mega-list
    matrices = [m for c in circuits for m in getCircuitMatrices2xE(c)]
    result = self.forward(matrices)
    # Result has shape=(n_qubits*sum(c.n_slices for c in circuits), circuit_emb_size), we need to
    # first split the slices of all circuits, then apply qubitwise maxpool to get slice embeddings
    # for each time slice. Then apply positional encoding and do slicewise maxpool to get a circuit
    # embedding for each time slice. Sigh.
    total_n_slices = sum(c.n_slices for c in circuits)
    result = result.reshape(total_n_slices, self.hw.n_physical_qubits, self.nn_dims[-1])
    slice_embs = torch.max(result, dim=1).values
    # Split the resulting tensor into a list with one tensor per circuit
    slice_embs = [slice_embs[i_ini:i_fi,:] for i_ini, i_fi in zip(indices[:-1], indices[1:])]
    # Add positional embeddings to slices of each circuit
    for n_slices, circuit_slice_embs in zip(slices_per_circuit, slice_embs):
      circuit_slice_embs += GNNEncoder.getPositionalEmbedding(n_slices, self.nn_dims[-1])
    circuit_embs = []
    # Each item in slice embs is a tensor. The ith row of this tensor corresponds to the embedding
    # of the ith time slice. We want a tensor in which the ith row contains a circuit embedding from
    # the ith time slice until the end (through pooling)
    for circuit in slice_embs:
      circuit_emb = torch.empty_like(circuit)
      for i in range(circuit.shape[0]):
        circuit_emb[i,:] = torch.max(circuit[i:,:], dim=0).values
      circuit_embs.append(circuit_emb)
    return circuit_embs, slice_embs