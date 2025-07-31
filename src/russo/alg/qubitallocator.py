import copy
import torch
import torch.nn as nn
from typing import Sequence, Tuple
from russo.alg.circuitencoder import CircuitSliceEncoder
from russo.alg.coresnapshotenc import CoreSnapshotEncoder
from russo.alg.decoder import Decoder



class QubitAllocator(nn.Module):
  ''' Entire pipeline of the circuit partitioning alg. (qubit allocation) from Ref. [1].

  Args:
    - num_lq: number of logical qubits.
    - emb_size: size of the embedding vectors (all of them).
    - num_enc_transf: number of layers of circuit encoder transformers.
    - num_enc_transf_heads: number of MHA heads used in each circuit encoder transformer.
    - core_con: matrix containing the core connectivity. Position (i,j), i != j, contains a 1 if
        core i is connected to core j, zero otherwise (no need of self-loops).
    - core_capacities: rank 1 tensor in which each element indicates the number of qubits its
        respective core can hold.

  References:
    [Attention-Based Deep Reinforcement Learning for Qubit Allocation in Modular Quantum Architectures]
    (https://arxiv.org/abs/2406.11452).
      Enrico Russo, Maurizio Palesi, Davide Patti, Giuseppe Ascia, Vincenzo Catania. 2024.
  '''
  
  def __init__(self, num_lq: int,
                     emb_size: int,
                     num_enc_transf: int,
                     num_enc_transf_heads: int,
                     core_con: torch.Tensor,
                     core_capacities: torch.Tensor,
  ):
    super().__init__()
    self.num_lq = num_lq
    self.register_buffer("core_con", copy.deepcopy(core_con))
    self.register_buffer("core_capacities", copy.deepcopy(core_capacities))
    self.qubit_embs = nn.Parameter(torch.randn(num_lq, emb_size), requires_grad=True)
    self.circuit_slice_encoder = CircuitSliceEncoder(emb_shape=emb_size,
                                                     num_enc_transf=num_enc_transf,
                                                     num_enc_transf_heads=num_enc_transf_heads)
    self.core_snapshot_encoder = CoreSnapshotEncoder(core_con=self.core_con,
                                                     core_emb_shape=emb_size)
    self.decoder = Decoder(core_capacities=self.core_capacities,
                           core_emb_size=emb_size,
                           slice_emb_size=emb_size)
  

  def _reorderQubits(self, adj: torch.Tensor) -> Sequence[Sequence[int]]:
    '''
    In the current slice, determines a slot (or qubit) ordering so that qubits interacting in gates appear first.

    Args:
    - adj: Tensor[B, Q, Q] adjency for this time slice t

    Returns:
    - perm: Tensor[B, Q] permutation indices for the slots
    - first_q: Tensor[B, Q] original qubit index in the slot
    - second_q: Tensor[B, Q] partner index or -1 for free qubits
    - is_pair: Tensor[B, Q] boolean mask indicating if the slot is a two-qubit gate
    '''
    device = adj.device
    B, Q = adj.shape[:2]

    # Mask of qubits involved in any gate
    in_gate = (adj.sum(dim=1) > 0).squeeze()  # [B, Q] mask
    # Natural qubit indices (from 0 to Q-1) for each batch
    idxs = torch.arange(Q, device=device).unsqueeze(0).expand(B, Q)  # [B, Q]
    # For each qubit in the slice, if it is in a gate, its key is idx, otherwise it is Q + idx
    keys = (~in_gate).to(torch.long) * Q + idxs   # [B, Q]
    # Puts all gate qubit slots first, then free qubits
    perm = keys.argsort(dim=1) 

    # Permute the index arrays
    first_q = torch.gather(idxs, 1, perm) # Which original qubit index is in the q-th slot after sorting
    partner = adj.argmax(-1) 
    # Get the partner qubit for each slot (-1 if none)
    second_q = torch.where(in_gate, partner, torch.full_like(partner, -1)) # [B, Q]
    second_q = torch.gather(second_q, 1, perm) # [B, Q]
    is_pair = (second_q >= 0)  # [B, Q]

    return perm, first_q, second_q, is_pair


  def _computeDistances(self, A_prev: torch.Tensor, second_q: torch.Tensor, is_pair: torch.BoolTensor) -> torch.Tensor:
      '''
      Computes the distances of the qubits in the current slice to the cores.

      Args:
      - A_prev: Tensor[B, Q] previous allocations for this time slice
      - second_q: Tensor[B, Q] partner index or -1 for free qubits
      - is_pair: Tensor[B, Q] boolean mask indicating if the slot is a two-qubit gate

      Returns:
      - distances: Tensor[B, Q, C] where C is the number of cores
      '''
      p0 = A_prev  # [B, Q]
      # Gather the previous allocation for the second qubit (if any)
      p1 = torch.gather(A_prev, 1, second_q.clamp(min=0))
      # Advanced indexing: for each batch and qubit, select the row in core_con corresponding
      # to the previous core assignment of the qubit. 
      d0 = self.core_con[p0]  # [B, Q, C]
      d1 = self.core_con[p1]  # [B, Q, C]
      # Total distance of qubits interacting in a gate to core c is the sum of individual distances
      distances = d0 + d1 * is_pair.unsqueeze(-1)

      return distances


  def _computeQubitEmbeddings(self, first_q: torch.Tensor, second_q: torch.Tensor, is_pair: torch.BoolTensor) -> torch.Tensor:
      '''
      Computes the qubit embeddings for the current slice.

      Args:
      - first_q: Tensor[B, Q] original qubit index in the slot
      - second_q: Tensor[B, Q] partner index or -1 for free qubits
      - is_pair: Tensor[B, Q] boolean mask indicating if the slot is a two-qubit gate

      Returns:
      - q_embs: Tensor[B, Q, d_E] where d_E is the embedding dimension
      '''
      emb1 = self.qubit_embs[first_q]  # [B, Q, d_E]
      emb2 = self.qubit_embs[second_q.clamp(min=0)]  # [B, Q, d_E]
      # Mask-average qubit embeddings only when there is a second qubit, otherwise use the first one
      q_embs = torch.where(is_pair.unsqueeze(-1), (emb1 + emb2) * 0.5, emb1)  # [B, Q, d_E]

      return q_embs

  def _updateAllocations(self, allocations: torch.Tensor, first_q_s: torch.Tensor, second_q_s: torch.Tensor, 
                        choice: torch.Tensor, is_pair_s: torch.BoolTensor, t: int) -> torch.Tensor:
      '''
      Updates the allocations tensor with the new choices made by the decoder.

      Args:
      - allocations: Tensor[B, Q, T] current allocations
      - first_q: Tensor[B] original qubit index in the slot
      - second_q: Tensor[B] partner index or -1 for free qubits
      - choice: Tensor[B] chosen core for each qubit
      - is_pair: Tensor[B] boolean mask indicating if the slot is a two-qubit gate
      - t: int current time slice index
      - s: int current qubit index in the slice

      Returns:
      - allocations: Updated allocations tensor
      '''
      # Mask to avoid double allocations
      assign_mask = (~is_pair_s) | (first_q_s < second_q_s)  # [B, Q]
      if not assign_mask.any():
          return allocations
      mask1 = assign_mask.nonzero(as_tuple=True)[0]
      allocations[mask1, first_q_s[mask1], t] = choice[mask1]

      # If it's a pair, update the second qubit as well
      mask2 = (is_pair_s & assign_mask).nonzero(as_tuple=True)[0]
      if mask2.numel():
          allocations[mask2, second_q_s[mask2], t] = choice[mask2]

      return allocations
  

  def getInvalidMask(self, core_capacities: torch.Tensor, double: bool, first_q: torch.Tensor, second_q: torch.Tensor,
                     allocations: torch.Tensor, t: int) -> torch.Tensor:
    ''' Implements two masks:
    1. For cores where there is not enough space for allocating the qubit. (Consideration 1 in the paper)
    2. For cores where an already allocated qubit of the current pair (if pair) wasn't allocated 
       in the current time slice. (Consideration 2 in the paper)
    For consideration 3 in the paper, the allocation is done in such a way that paired qubits are always allocated
    first than free qubits, so a mask is not needed. 
    
    Args:
      - core_capacities: rank 1 tensor that contains the number of qubits that can be held per core.
      - double: whether this decoding corresponds to two qubits that act in a gate.
      - first_q: original qubit index in the slot.
      - second_q: partner index or -1 for free qubits.

    Returns:
      - invalid_mask: boolean mask indicating invalid cores (True if invalid, False if valid).

    '''
    batch_indices = torch.arange(allocations.shape[0], device=allocations.device)
    is_assigned = double & (first_q > second_q) 
    core_indices = allocations[batch_indices, second_q, t] # [B]

    mask_cap = core_capacities < torch.where(double.unsqueeze(-1), 2, 1) # [B, C]
    mask_nonfriends = torch.zeros_like((core_capacities), dtype=torch.bool) # [B, C]
    # Undo the capacity mask if the qubit was already assigned, as we already removed capacity=2

    if is_assigned.any():
        b = is_assigned.nonzero(as_tuple=True)[0]
        mask_cap[b, core_indices[b]] = False

        mask_nonfriends = torch.ones_like(core_capacities, dtype=torch.bool) # [B, C]
        # Now mask all cores where the current qubit's pair was assigned 
        mask_nonfriends[batch_indices, core_indices] = False 
        # mask[~is_assigned, :] = False  # Unmask if previously unassigned
        mask_nonfriends = mask_nonfriends & is_assigned.unsqueeze(1)

    return mask_nonfriends | mask_cap  # [B, C]

  
  def forward(self, circuit_slice_gates: Tuple[Tuple[int,int], ...],
                    circuit_slice_matrices: torch.Tensor,
                    greedy: bool):
    B, T, Q, _ = circuit_slice_matrices.shape
    assert (self.num_lq == Q), \
            "matrix shape in circuit_slice_matrices does not match number of logical qubits"
    assert (len(circuit_slice_gates[0]) == T), \
            "length of circuit_slice_gates does not match length of circuit_slice_matrices"
    device = next(self.parameters()).device
    allocations = torch.zeros(size=(B, self.num_lq, T), dtype=int, device=device)
    H_S, H_X = self.circuit_slice_encoder(circuit_slice_matrices, self.qubit_embs)
    all_log_probs = []

    for t in range(T):
      # Allocations is initially filled with 0, so at first iteration column 0 is fine.
      A_prev = allocations[:, :, max(0,t-1)].squeeze()
      Ht_C = self.core_snapshot_encoder(A_prev, self.qubit_embs).squeeze(1) # [B, C, d_H] where T = 1

      adj = circuit_slice_matrices[:, t]
      perm, first_q, second_q, is_pair = self._reorderQubits(adj)  # [B, Q], [B, Q], [B, Q], [B, Q]
      A_prev = torch.gather(A_prev, 1, perm)  # [B, Q] - reordering previous allocations

      q_embs = self._computeQubitEmbeddings(first_q, second_q, is_pair)  # [B, Q, d_E]
      if t == 0:
          distances = torch.zeros((B, Q, self.core_capacities.shape[0]), device=device)
      else:
          distances = self._computeDistances(A_prev, second_q, is_pair)  # [B, Q, C]
      HS_t = H_S[:, t]  # [B, d_H]

      # Reset capacities for this slice
      core_capacities = self.core_capacities.unsqueeze(0).expand(B, -1).detach().clone()  # [B, C]

      for s in range(Q):
        emb_s = q_embs[:, s] # [B, emb]
        dist_s = distances[:, s] # [B, C]
        pair_s = is_pair[:, s] # [B] bool
        first_q_s = first_q[:, s] # [B]
        second_q_s = second_q[:, s] # [B]

        invalid_mask = self.getInvalidMask(core_capacities, pair_s, first_q_s, second_q_s, allocations, t)

        probs = self.decoder(Ht_C, core_capacities.clone(), dist_s, H_X, HS_t, emb_s, pair_s, invalid_mask)  # [B, C]
        cat = torch.distributions.Categorical(probs=probs)
        choice = cat.probs.argmax(1) if greedy else cat.sample()
        all_log_probs.append(cat.log_prob(choice))

        allocations = self._updateAllocations(allocations, first_q_s, second_q_s, choice, pair_s, t)
        
        sub_mask = (pair_s & (first_q_s < second_q_s)) | ~pair_s
        if sub_mask.any():
            # For pairs, only decrement caps if no qubit in the pair was already assigned
            valid_idx = sub_mask.nonzero(as_tuple=True)[0]
            core_capacities[torch.arange(B, device=device)[valid_idx], choice[valid_idx]] -= (1 + pair_s[valid_idx].long())

    # This is just for testing
    cores = torch.arange(self.core_capacities.shape[0], device=allocations.device)
    ok = (allocations.unsqueeze(-1) == cores).sum(1) <= self.core_capacities
    assert ok.all(), "Invalid allocation: some cores overfilled"
      
    return allocations, torch.stack(all_log_probs, dim=1)  # [B, Q, T], [B, Q*T]



