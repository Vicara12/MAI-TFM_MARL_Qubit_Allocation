import math
from typing import Tuple
import torch
import torch.nn as nn
from models.gnn import GNN


class CircuitEncoder(nn.Module):
  ''' Given a circuit returns the slice embeddings H^(S) and circuit embeddings H^(X).

  This class implements section III B from Ref. [1].

  Args:
    num_lq: number of logical qubits in the circuit.
    circuit_slices: adjacency matrix of gate connections for each time slice in the circuit.
    slice_emb_shape: shape of the time slice embeddings, d_E in Ref. [1].
    circuit_emb_shape: shape of the whole circuit embedding, d_H in Ref. [1].
    num_enc_transf: number of transformer blocks for the encoder, b in Ref. [1].

  References:
    [Attention-Based Deep Reinforcement Learning for Qubit Allocation in Modular Quantum Architectures]
    (https://arxiv.org/abs/2406.11452).
      Enrico Russo, Maurizio Palesi, Davide Patti, Giuseppe Ascia, Vincenzo Catania. 2024.

  '''

  @classmethod
  def getPositionalEmbedding(T, d_model):
    position = torch.arange(T).unsqueeze(1)  # [T, 1]
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(T, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # [T, d_model]

  def __init__(self,
               circuit_slices: Tuple[torch.ndarray],
               slice_emb_shape: int,
               circuit_emb_shape: int,
               num_enc_transf: int
      ):
    super().__init__()
    T = len(circuit_slices)
    num_lq = circuit_slices[0].shape[0] # Number of (logical) qubits in the circuit
    self.q_embeddings = nn.Parameter(torch.randn(num_lq, slice_emb_shape))
    self.gnn = GNN(graphs=circuit_slices, out_shape=slice_emb_shape)
    self.pos_emb = CircuitEncoder.getPositionalEmbedding(T, slice_emb_shape)
    self.register_buffer("positional_embedding", self.pos_enc)
  
  def forward(self):
    Ht_IQ = self.gnn(self.q_embeddings) # H_t^{(I,Q)} shape = [T, Q, d_E]
    Ht_I = torch.max(Ht_IQ, dim=1)      # Max pool across qubit dimension, shape = [T, d_E]
    Ht_I += self.pos_emb