from typing import Optional, Tuple
import torch
import torch.nn as nn

from .encoder import CoreFeatureEncoder, DynamicAgentGrouper
from .nn.transformer import (
    TransformerBlock as CommunicationLayer, 
    Normalization
)

class QAContextEmbedding(nn.Module):
    """Context embeddings.

    Produces:
      - Agent (qubit) embeddings (slice-aware)
      - Core embeddings (snapshot + features)
      - Temporal/global tokens
    """
    def __init__(
        self,
        embed_dim: int,
        use_communication: bool = True,
        num_communication_layers: int = 1,
        use_final_norm: bool = False,
        grouper: Optional[DynamicAgentGrouper] = None,
        grouper_kwargs: Optional[dict] = None,
        **communication_layer_kwargs,
    ):
        super().__init__()

        # Agents embeddings (both global and contextual) are precomputed in initial embedding
        # so we've already called those, now we just need to retrieve the index we need
        grouper_kwargs = grouper_kwargs or {}
        layer_kwargs = dict(communication_layer_kwargs)
        
        if grouper is None:
            self.grouper = DynamicAgentGrouper(
                embed_dim=embed_dim, 
                **grouper_kwargs
            )
        else:
            self.grouper = grouper

        self.core_feature_enc = CoreFeatureEncoder(
            embed_dim=embed_dim,
        )
        
        self.use_communication = use_communication
        if self.use_communication:
            self.q_layers = nn.Sequential(
                *(
                    CommunicationLayer(embed_dim=embed_dim, **layer_kwargs)
                    for _ in range(num_communication_layers)
                )
            )
            self.c_layers = nn.Sequential(
                *(
                    CommunicationLayer(embed_dim=embed_dim, **layer_kwargs)
                    for _ in range(num_communication_layers)
                )
            )
        else:
            self.q_layers = nn.Identity()
            self.c_layers = nn.Identity()

        self.norm = (
            Normalization(embed_dim, layer_kwargs.get("normalization", "rms"))
            if use_final_norm
            else None
        )

        self.project_global = nn.Linear(2*embed_dim, embed_dim)

    def _agent_state_embedding(self, embeddings, global_embeddings=None, **kwargs):
        agent_slice_embeds = embeddings  # already [B, Q, d]

        if global_embeddings is not None:
            agent_embds = torch.cat([agent_slice_embeds, global_embeddings], dim=-1)
            return self.project_global(agent_embds)

        return agent_slice_embeds
    
    def _agent_global_embedding(self, embeddings, **kwargs):
        raise NotImplementedError #TODO: See if we can avoid using this

    def forward(
            self, 
            slice_embds: torch.Tensor,
            prev_core_allocs: torch.Tensor,
            current_core_allocs: torch.Tensor,
            core_capacities: torch.Tensor,
            core_size: torch.Tensor,
            core_connectivity: torch.Tensor = None,
            adj_matrix: torch.Tensor = None,
            action_mask: torch.Tensor = None,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # Gather the agent embeddings
        agent_embeds = self._agent_state_embedding(slice_embds) 

        # Agent embeddings
        grouped_agent_emds, agent_mask, agent_demands, final_action_mask = self.grouper(
            agent_embeds, 
            prev_core_allocs=prev_core_allocs,
            current_core_allocs=current_core_allocs,
            core_connectivity=core_connectivity,
            adj_matrix=adj_matrix,
            action_mask=action_mask
            )  # [Q, d] -> [Agents, C, d]

        # Gather core embeddings (capacity)
        core_embeds = self.core_feature_enc(core_capacities, core_size=core_size)  # [C, d]

        agent_embeds = grouped_agent_emds + core_embeds  # [Agents, C, d]

        if self.use_communication:
            if agent_embeds.dim() == 3:
                _, C, _ = agent_embeds.shape

                # [Agents, C, d] -> [C, Agents, d]
                q_view = agent_embeds.permute(1, 0, 2)
                # we must mask the padding agents so they don't participate!
                q_mask = agent_mask.unsqueeze(0).repeat_interleave(C, dim=0) # [Agents] -> [C, Agents]
                padding_mask = ~q_mask
                for layer in self.q_layers:
                    q_view = layer(q_view, mask=padding_mask)
                # [C, Agents, d] -> [Agents, C, d]
                agent_embeds = q_view.permute(1, 0, 2)

                # [Agents, C, d]
                agent_embeds = self.c_layers(agent_embeds)

        if self.norm is not None:
            # TODO: Make sure batchnorm isn't used here
            agent_embeds = self.norm(agent_embeds)

        return agent_embeds, agent_mask, agent_demands, final_action_mask