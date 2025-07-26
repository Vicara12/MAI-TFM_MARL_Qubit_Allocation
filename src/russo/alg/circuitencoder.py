import math
from typing import Tuple
import torch
import torch.nn as nn

from russo.models.gnn import GNN
from russo.models.transformer import TransformerEncoder


class CircuitSliceEncoder(nn.Module):
  ''' Given a circuit returns the slice embeddings H^(S) and circuit embeddings H^(X).

  This class implements section III B 1 & 2 from Ref. [1].

  Args:
    - emb_shape: shape of the time slice embeddings, d_E in Ref. [1].
    - num_enc_transf: number of transformer blocks for the encoder, b in Ref. [1].
    - num_enc_transf_heads: number of transformer heads per block in the encoder, b in Ref. [1].

  References:
    [Attention-Based Deep Reinforcement Learning for Qubit Allocation in Modular Quantum Architectures]
    (https://arxiv.org/abs/2406.11452).
      Enrico Russo, Maurizio Palesi, Davide Patti, Giuseppe Ascia, Vincenzo Catania. 2024.
  '''

  @staticmethod
  def getPositionalEmbedding(T, d_model):
    position = torch.arange(T).unsqueeze(1)  # [T, 1]
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(T, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # [T, d_model]

  def __init__(self,
               emb_shape: int,
               num_enc_transf: int,
               num_enc_transf_heads: int
      ):
    super().__init__()
    self.emb_shape = emb_shape
    self.gnn = GNN(emb_shape=emb_shape)
    self.enc_transf = TransformerEncoder(num_layers=num_enc_transf,
                                         embed_dim=emb_shape,
                                         num_heads=num_enc_transf_heads,
                                         ff_hiden_dim=emb_shape,
                                         dropout=0.0)
  
  def forward(self, circuit_slices: Tuple[torch.Tensor], q_embeddings: torch.Tensor) -> torch.Tensor:
    '''
    Args:
      - circuit_slices: adjacency matrix of gate connections for each time slice in the circuit.
      - q_embeddings shape: [Q, d_E] where Q = number of logical qubits, d_E slice emb. dim.
    Returns:
      - H_S: encoded slice embeddings of shape: [T, d_H] where T = num. slices and d_H = d_E
      - H_X: circuit representation of shape d_H = d_E
    '''
    device = next(self.parameters()).device
    T = len(circuit_slices)
    self.gnn.setGraphs(graphs=circuit_slices)
    # Section III. B.1 InitEmbedding
    Ht_IQ = self.gnn(q_embeddings) # H_t^{(I,Q)} shape = [T, Q, d_E]
    Ht_I, _ = torch.max(Ht_IQ, dim=1)      # Max pool across qubit dimension, shape = [T, d_E]
    Ht_I += CircuitSliceEncoder.getPositionalEmbedding(T, self.emb_shape).to(device)
    # Section III. B.2 EncoderBlocks
    H_S = self.enc_transf(Ht_I)  # shape = [T, d_E] (we force d_E = d_H)
    H_X = torch.mean(H_S, dim=0) # Circuit embedding, shape = [d_E]
    return H_S, H_X