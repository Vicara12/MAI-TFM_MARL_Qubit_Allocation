from typing import NamedTuple, Optional, Tuple
from src.qalloczero.models.encoder import DynamicAgentGrouper, QubitContextTransformer, QubitEmbedding, QAInitEmbedding
from src.qalloczero.models.context_embedding import QAContextEmbedding
from src.qalloczero.models.decoder import QADecoder
import torch

from src.utils.conflict_handler import AGENT_HANDLER_REGISTRY, AgentHandler, HighestProbabilityAgentHandler, NoHandler


class PredictionOutputs(NamedTuple):
    probs: Optional[torch.Tensor]
    logits: Optional[torch.Tensor]
    log_probs: torch.Tensor
    final_mask: torch.Tensor
    agent_demands: Optional[torch.Tensor]


class PredictionModel(torch.nn.Module):
  ''' Model used to predict the most likely core allocation for a given state.

  This model is used to predict the most likely core allocation for a given state, which is then
  used as input for the value function predictor. The model is trained with cross entropy loss on
  the core allocation predicted by the policy model.
  '''

  def __init__(
    self,
    embed_size: int,
    circuit_embds_kwargs = {},
    context_embds_kwargs = {},
    decoder_kwargs = {},
  ):
    super().__init__()
    #TODO: Change number of qubits to max num qubits (for the embeddings) 
    self.circuit_embds = QAInitEmbedding(embed_dim=embed_size, num_qubits=20, **circuit_embds_kwargs)  
    self.grouper = DynamicAgentGrouper(embed_dim=embed_size)
    self.context_embds = QAContextEmbedding(embed_dim=embed_size, grouper=self.grouper, **context_embds_kwargs)
    self.decoder = QADecoder(embed_dim=embed_size, **decoder_kwargs)
    
    self.output_logits_ = False
    self.output_demands_ = False
    
  def set_dropout(self, p: float):
    for module in self.modules():
        # 1. Update standard Dropout layers (in Sequential & Transformers)
        if isinstance(module, torch.nn.Dropout):
            module.p = p
        # 2. Update MultiheadAttention layers (attribute based)
        if isinstance(module, torch.nn.MultiheadAttention):
            module.dropout = p
  
  def output_logits(self, value: bool):
    self.output_logits_ = value

  def output_demands(self, value: bool):
    self.output_demands_ = value
  
  def get_circuit_embds(self, adj_matrices: torch.Tensor) -> torch.Tensor:
    return self.circuit_embds(adj_matrices)

  def forward(
      self,
      slice_embds: torch.Tensor,
      prev_core_allocs: torch.Tensor,
      current_core_allocs: torch.Tensor,
      core_capacities: torch.Tensor,
      core_size: torch.Tensor,
      core_connectivity: torch.Tensor,
      adj_matrix: torch.Tensor,
      action_mask: torch.Tensor,
  ) -> PredictionOutputs:

    agent_embeds, agent_mask, agent_demands, final_action_mask = self.context_embds(
      slice_embds,
      prev_core_allocs,
      current_core_allocs,
      core_capacities,
      core_size,
      core_connectivity,
      adj_matrix,
      action_mask,
    )

    logits, final_mask = self.decoder(agent_embeds, final_action_mask, agent_mask)
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1) 

    return PredictionOutputs(
      probs=probs,
      logits=logits if self.output_logits_ else None,
      log_probs=log_probs,
      final_mask=final_mask,
      agent_demands=agent_demands if self.output_demands_ else None,
  )

