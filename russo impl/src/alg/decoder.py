import math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
  ''' Decoder for the autoregressive qubit allocation core assignment.

  The decoding process should always be applied first for the qubits that belong to a gate in the
  given time slice. After that all other "lone" qubits can be allocated. Otherwise the impossible
  allocation condition described in the last paragraph of section III E from Ref. [1] could happen.
  
  When using forward with qubits that do not belong to a quantum gate the qubit embedding Eq_Q is
  set to that individual qubit and double is to be set to False. When using forward with a pair
  of qubits that belong to a quantum gate double is to be set to True and the qubit embedding must
  be a mix of the embeddings of both qubits (avg or mix, for example). The returned core allocation
  is for both qubits in the gate.

  Args:
    - core_capacities: list that contains the number of qubits that can be held per core.
    - core_emb_size: length of the output core embedding, d_H in Ref. [1].
    - slice_emb_size: length of the slice embedding, d_E in Ref. [1].

  References:
    [Attention-Based Deep Reinforcement Learning for Qubit Allocation in Modular Quantum Architectures]
    (https://arxiv.org/abs/2406.11452).
      Enrico Russo, Maurizio Palesi, Davide Patti, Giuseppe Ascia, Vincenzo Catania. 2024.
  '''
  def __init__(self, core_capacities: List[int], core_emb_size: int, slice_emb_size: int):
    super().__init__()
    self.slice_emb_size = slice_emb_size
    self.capacity_emb = nn.Embedding(max(core_capacities)+1, core_emb_size)
    self.dist_emb = nn.Embedding(2, core_emb_size) # I'll keep distance binary for now (connected/not connected)
    self.W_q = nn.Linear(3*slice_emb_size, core_emb_size, bias=False)
    self.W_k = nn.Linear(slice_emb_size, slice_emb_size, bias=False)
    self.W_v = nn.Linear(slice_emb_size, slice_emb_size, bias=False)
  
  def _getDynamicCoreEmbeddings(self, Ht_C: torch.Tensor, core_capacities: torch.Tensor,
                                distances: torch.Tensor) -> torch.Tensor:
    ''' Get the dynamic core embeddings G_{t,q}^{(C)}.
    '''
    return Ht_C + self.capacity_emb(core_capacities) + self.dist_emb(distances)
  
  def _getInvalidMask(self, core_capacities: torch.Tensor, double: bool) -> torch.Tensor:
    ''' Return True for cores where there is not enough space for allocating the qubit.
    '''
    if double:
      return (core_capacities < 2)
    return (core_capacities == 0)
  
  def forward(self, Ht_C: torch.Tensor, core_capacities: torch.Tensor, distances: torch.Tensor,
              H_X: torch.Tensor, Ht_S: torch.Tensor, Eq_Q: torch.Tensor, double: bool) -> torch.Tensor:
    '''
    Args:
      - Ht_C: core embeddings of shape: [C, d_H]
      - core_capacities: remaining unallocated qubits for each core. Shape: [C].
      - distances: distances from the previous allocation of qubit q to all other cores. Shape: [C].
      - H_X: circuit embedding. Shape: [d_H].
      - Ht_S: t-th circuit slice embedding. Shape: [d_H].
      - Eq_Q: q-th qubit embedding. Shape: [d_H].
      - double: wether this decoding corresponds to two qubits that act in a gate.
    Returns:
      - Vector with the probabilities of allocating qubit q to each core. Shape: [C].
    '''
    sqrt_dH = math.sqrt(self.slice_emb_size)
    Gtq_C = self._getDynamicCoreEmbeddings(Ht_C, core_capacities, distances) # [C, d_H]
    # Apply pointer attention mechanism
    context = torch.cat([H_X, Ht_S, Eq_Q], dim=-1) # [3*d_H]
    Q = self.W_q(context) # [d_H]
    K = self.W_k(Gtq_C)   # [C, d_H]
    V = self.W_v(Gtq_C)   # [C, d_H]
    attn_logits = torch.matmul(K,Q)/sqrt_dH # [C, d_H]x[d_H] -> [C]
    attn_weights = F.softmax(attn_logits, dim=0) # [C]
    glimpse = torch.matmul(attn_weights, V)      # [C]x[C, d_H] -> [d_H]
    # Compute scores and mask invalid qubits
    invalid_mask = self._getInvalidMask(core_capacities, double)
    u_tqc = torch.matmul(K, glimpse)/sqrt_dH # [C, d_H]x[d_H] -> [C]
    u_tqc = u_tqc.masked_fill(invalid_mask, float('-inf')) # [C]
    return F.softmax(u_tqc, dim=0) # [C]