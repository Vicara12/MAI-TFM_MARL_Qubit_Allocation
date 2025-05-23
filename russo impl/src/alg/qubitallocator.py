import torch
import torch.nn as nn
from typing import Tuple
from alg.circuitencoder import CircuitSliceEncoder
from alg.coresnapshotenc import CoreSnapshotEncoder
from alg.decoder import Decoder



class QubitAllocator(nn.Module):
  ''' Entire pipeline of the circuit partitioning alg. (qubit allocation) from Ref. [1].

  Args:
    - num_lq: number of logical qubits.
    - emb_size: size of the embedding vectors (all of them).
    - num_enc_transf: number of layers of circuit encoder transformers.
    - num_enc_transf_heads: number of MHA heads used in each circuit encoder transformer.
    - core_con: matrix containing the core connectivity. Position (i,j), i != j, contains a 1 if
        core i is connected to core j, zero otherwise (no need of self-loops).
    - core_capacities: tuple in which each element indicates the number of qubits its respective
        core can hold.

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
                     core_capacities: Tuple[int, ...]):
    self.num_lq = num_lq
    self.core_capacities = core_capacities
    self.core_con = core_con
    self.qubit_embs = nn.Parameter(torch.randn(num_lq, emb_size), requires_grad=True)
    self.circuit_slice_encoder = CircuitSliceEncoder(num_lq=num_lq,
                                                     emb_shape=emb_size,
                                                     num_enc_transf=num_enc_transf,
                                                     num_enc_transf_heads=num_enc_transf_heads)
    self.core_snapshot_encoder = CoreSnapshotEncoder(core_con=core_con,
                                                     core_emb_shape=emb_size)
    self.decoder = Decoder(core_capacities=core_capacities,
                           core_emb_size=emb_size,
                           slice_emb_size=emb_size)
  

  @classmethod
  def _getOrderedQubits(gates: Tuple[Tuple[int,int], ...], num_lq: int):
    ''' Returns the qubits in the correct order in order to be fed to the decoder.

    The qubits are returned as a tuple of tuples. Each item contains either two elements for the
    qubits that belong to a gate in this time slice or a single element when the qubits are free.
    Pairs of qubits that form gates are returned first, followed by free qubits.
    '''
    qubits_in_gates = set(qubit for gate in gates for qubit in gate)
    qubits_not_in_gates = set(range(num_lq)) - qubits_in_gates
    return gates + tuple((qubit,) for qubit in qubits_not_in_gates)


  def forward(self, circuit_slice_gates: Tuple[Tuple[int,int], ...],
                    circuit_slice_matrices: Tuple[torch.Tensor],
                    greedy: bool):
    assert (self.num_lq == circuit_slice_matrices[0].shape[0]), \
            "matrix shape in circuit_slice_matrices does not match number of logical qubits"
    assert (len(circuit_slice_gates) == len(circuit_slice_matrices)), \
            "length of circuit_slice_gates does not match length of circuit_slice_matrices"
    num_slices = len(circuit_slice_matrices)
    allocations = torch.zeros(size=(self.num_lq, num_slices))
    H_S, H_X = self.circuit_slice_encoder(circuit_slice_matrices, self.qubit_embs)
    all_log_probs = []
    for t, slice_gates in enumerate(circuit_slice_gates):
      # allocations is initially filled with -1, so at first iteration column 0 is fine.
      A_prev = allocations[:,max(0,t-1)].squeeze()
      Ht_C = self.core_snapshot_encoder(A_prev, self.qubit_embs)
      core_capacities = torch.tensor(self.core_capacities)
      for q_tuple in QubitAllocator._getOrderedQubits(slice_gates, self.num_lq):
        # Get qubit embedding if 1 qubit or mean of both if gate
        q_embs = self.qubit_embs[q_tuple,:].mean(dim=0)
        if t == 0:
          distances = torch.zeros(size=(len(self.core_capacities),))
        else:
          prev_cores = A_prev[q_tuple,] # For each qubit in q_tuple get its previous core allocation
          # Total distance of qubits in q_tuple to core c is the sum of individual distances
          distances = self.core_con[prev_cores,:].sum(dim=0)
        double = (len(q_tuple) == 2)
        core_probs = self.decoder(Ht_C, core_capacities, distances, H_X, H_S[t], q_embs, double)
        core_probs = torch.distributions.Categorical(core_probs)
        # Select core from distribution and update allocations matrix and core capacities
        core = core_probs.max() if greedy else core_probs.sample()
        allocations[q_tuple,t] = core
        core_capacities[core] -= len(q_tuple)
        all_log_probs.append(core_probs.log_prob(core))
    return allocations, all_log_probs