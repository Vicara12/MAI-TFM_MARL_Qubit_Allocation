import torch
import torch.nn as nn

from models.gnn import GNN


class CoreSnapshotEncoder(nn.Module):
  ''' Given last core assignments and qubit embeddings generate an embedding of previous qubit allocations.

  This class implements section III C from Ref. [1].

  Args:
    - core_con: matrix of core connectivities of the architecture.
    - core_emb_shape: length of the output core embedding, d_H in Ref. [1].

  References:
    [Attention-Based Deep Reinforcement Learning for Qubit Allocation in Modular Quantum Architectures]
    (https://arxiv.org/abs/2406.11452).
      Enrico Russo, Maurizio Palesi, Davide Patti, Giuseppe Ascia, Vincenzo Catania. 2024.
  '''

  def __init__(self, core_con: torch.Tensor, core_emb_shape: int):
    super().__init__()
    self.n_cores = core_con.shape[0]
    self.padding_emb = nn.Parameter(torch.zeros((core_emb_shape,)))
    self.gnn = GNN(emb_shape=core_emb_shape)
    self.gnn.setGraphs(graphs=core_con.unsqueeze(0).float()) # Add batch dim at the front
  
  def forward(self, prev_assign: torch.Tensor, q_embeddings: torch.Tensor) -> torch.Tensor:
    '''
    Args:
      - prev_assign: for each logical qubit, the core to which it has been mapped. Shape: [Q]
      - q_embeddings shape: [Q, d_E] where Q = number of logical qubits, d_E slice emb. dim.
    Returns:
      - Core embeddings of shape [C, d_H] where C = number of cores, d_H = embedding size.
    '''
    core_embs = []
    for C in range(self.n_cores):
      mask = (prev_assign == C)
      if mask.any():
        core_embs.append(q_embeddings[mask].max(dim=0)[0]) # Take max pool of all q embs. in core C
      else:
        core_embs.append(self.padding_emb) # If no qubits in core C append learnable padding emb.
    core_embs = torch.stack(core_embs).to(q_embeddings.device)
    return self.gnn(core_embs).squeeze() # Transform core embs through GNN and remove "batch" dim