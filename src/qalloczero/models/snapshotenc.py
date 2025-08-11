from typing import Tuple, List
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
from utils.customtypes import Hardware


class SnapEncModel(torch.nn.Module):
  def __init__(self,
               nn_dims: Tuple[int],
               hardware: Hardware,
               qubit_embs: torch.Tensor,
               dummy_qubit_emb: torch.Tensor):
    super().__init__()
    self.hw = hardware
    self.qemb_len = qubit_embs.shape[1]
    self.qubit_embs = qubit_embs
    self.dummy_qubit_emb = dummy_qubit_emb.unsqueeze(0)
    self.convs = torch.nn.ModuleList()
    for in_dim, out_dim in zip(nn_dims[:-1], nn_dims[1:]):
      self.convs.append(GCNConv(in_dim, out_dim))


  def gnn(self, core_embs: List[torch.Tensor]) -> torch.Tensor:
    # TODO: check zeros
    batch = Batch.from_data_list(
      [Data(x=ce,
            edge_index=self.hw.sparse_core_con,
            edge_weight=self.hw.sparse_core_ws) for ce in core_embs])
    x = batch.x
    for conv in self.convs:
      x = conv(x, batch.edge_index, batch.edge_weight)
      x = torch.relu(x)
    return x


  def unmixedCoreEmbs(self, prev_core_allocs: torch.Tensor) -> torch.Tensor:
    ''' Get the core embeddings before the GNN with the core adjacency matrix
    '''
    core_embs = torch.empty(size=(self.hw.n_cores, self.qemb_len))
    for (core_i, core_cap) in enumerate(self.hw.core_capacities):
      qubits_in_core = (prev_core_allocs == core_i).nonzero().flatten()
      core_qubits_tensor = torch.empty(size=(core_cap, self.qemb_len))
      n_alloc = len(qubits_in_core)
      if n_alloc != 0:
        core_qubits_tensor[:n_alloc, :] = self.qubit_embs[qubits_in_core]
      # Fill remainder of qubit emb with repetitions of the dummy qubit
      core_qubits_tensor[n_alloc:, :] = self.dummy_qubit_emb.expand(core_cap-n_alloc,-1)
      core_embs[core_i,:] = torch.max(core_qubits_tensor, dim=0).values
    return core_embs

  
  def forward(self, core_allocs: List[torch.Tensor]) -> List[torch.Tensor]:
    pregnn_embs = []
    for prev_alloc in core_allocs:
      pregnn_embs.append(self.unmixedCoreEmbs(prev_alloc))
    result = self.gnn(pregnn_embs)
    core_embs = []
    n_rows = result.shape[0]
    nc = self.hw.n_cores
    for idx_from, idx_to in zip(range(0,n_rows, nc), range(nc, n_rows+1, nc)):
      core_embs.append(result[idx_from:idx_to])
    return core_embs