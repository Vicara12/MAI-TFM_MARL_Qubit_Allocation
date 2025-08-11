from typing import Tuple, Optional
from dataclasses import dataclass
import torch
from math import sqrt
from utils.customtypes import Hardware, GateType


class PredictionModel(torch.nn.Module):
  
  @dataclass
  class Config:
    hw: Hardware
    circuit_emb_shape: int
    mha_num_heads: int = 4


  def __init__(self,
               config: Config,
               qubit_embs: torch.Tensor):
    super().__init__()
    self.cfg = config
    self.qemb_len = qubit_embs.shape[1]
    self.qubit_embs = qubit_embs
    ce_len = self.cfg.circuit_emb_shape
    self.cap_emb_nn = self.__getEmbNN(ce_len)
    self.dist_emb_nn = self.__getEmbNN(ce_len)
    self.context_nn = torch.nn.Linear(2*ce_len + self.qemb_len, ce_len)
    self.mha = torch.nn.MultiheadAttention(embed_dim=ce_len,
                                           num_heads=self.cfg.mha_num_heads,
                                           batch_first=True)
    self.softmax = torch.nn.Softmax(dim=-1)
    self.glimpse_proj_nn = torch.nn.Sequential(
      torch.nn.Linear(ce_len,   ce_len//2),
      torch.nn.ReLU(),
      torch.nn.Linear(ce_len//2, ce_len//4),
      torch.nn.ReLU(),
      torch.nn.Linear(ce_len//4, 1)
    )


  def __getEmbNN(self, emb_len) -> torch.nn.Module:
    return torch.nn.Sequential(
      torch.nn.Linear(1, emb_len//2),
      torch.nn.ReLU(),
      torch.nn.Linear(emb_len//2, emb_len)
    )
  

  def __recomputeK(self, x: torch.Tensor) -> torch.Tensor:
    E = self.mha.embed_dim
    w_k = self.mha.in_proj_weight[E:2*E, :]
    b_k = self.mha.in_proj_bias[E:2*E]
    return torch.nn.functional.linear(x, w_k, b_k)


  def forward(self,
              current_alloc: Tuple[int, GateType],
              core_embs: torch.Tensor,
              prev_core_allocs: Optional[torch.Tensor],
              current_core_capacities: torch.Tensor,
              circuit_emb: torch.Tensor,
              slice_emb: torch.Tensor
              ) -> Tuple[torch.Tensor, float]:
    (_, gate) = current_alloc
    qubit_emb = torch.mean(self.qubit_embs[current_alloc[1],:], dim=0)
    context = self.context_nn(torch.concat([circuit_emb, slice_emb, qubit_emb], dim=-1)).unsqueeze(0)
    if prev_core_allocs is not None:
      prev_cores = prev_core_allocs[gate,]
      distances = torch.sum(self.cfg.hw.core_connectivity[prev_cores,:], dim=0)
    else:
      distances = torch.zeros_like(self.cfg.hw.core_capacities)
    g_tq = core_embs + \
          self.cap_emb_nn(current_core_capacities.float().unsqueeze(1)) + \
          self.dist_emb_nn(distances.float().unsqueeze(1))
    glimpse, _ = self.mha(context, g_tq, g_tq)

    # Get probabilities
    K = self.__recomputeK(g_tq)
    sqrt_dk = sqrt(self.qemb_len)
    u = torch.mm(glimpse/sqrt_dk, K.T)
    # Illegal actions are not taken into account here as this is suppressed in the MCTS call
    probs = self.softmax(u)

    # Get values
    v = self.glimpse_proj_nn(glimpse.squeeze())
    return probs.squeeze(), v