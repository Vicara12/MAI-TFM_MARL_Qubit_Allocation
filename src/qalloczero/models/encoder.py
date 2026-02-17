from typing import Optional

import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from .nn.positional_encoder import PositionalEncoder
from .nn.transformer import TransformerBlock


_ORTHOGONAL_IDS_CACHE = {}



class QubitEmbedding(nn.Module):
    """
    Embeds each qubit (agent) in each slice using a GNN.
    Output: [batch, num_slices, num_qubits, hidden_dim]
    """
    def __init__(self, embed_size, max_qubits, dropout=0.0, use_learnable_ids=False, use_temp_transformer=True):
        super().__init__()
        self.embed_size = embed_size
        self.max_qubits = max_qubits
        self.use_learnable_ids = use_learnable_ids 
        self.use_temp_transformer = use_temp_transformer
        
        if use_learnable_ids:
            # Original learnable embeddings
            self.qubit_ids = nn.Embedding(self.max_qubits+1, self.embed_size, padding_idx=self.max_qubits)
        else:
            self.register_buffer("qubit_ids", self._create_ids(max_qubits, embed_size))
            
        self.slice_encoding = gnn.GCNConv(self.embed_size, self.embed_size, add_self_loops=False)
        self.positional_encoding = PositionalEncoder(self.embed_size)
    
    def _create_ids(self, max_qubits, embed_size):
        cache_key = (max_qubits, embed_size)
        
        if cache_key not in _ORTHOGONAL_IDS_CACHE:
            if max_qubits > embed_size:
                raise ValueError(f"Cannot create {max_qubits} orthogonal vectors in {embed_size}D space")
            #TODO: Relax this constraint in future? (for more qubits)
            
            generator = torch.Generator()
            generator.manual_seed(42)  # Fixed seed!!
            
            random_matrix = torch.randn(embed_size, max_qubits, generator=generator)
            q, _ = torch.linalg.qr(random_matrix, mode='reduced')
            
            ids = q[:, :max_qubits].t()
            ids = ids / ids.norm(dim=1, keepdim=True)
            
            # We're caching the tensor so as to ensure consistency across all instances
            _ORTHOGONAL_IDS_CACHE[cache_key] = ids
        
        return _ORTHOGONAL_IDS_CACHE[cache_key].clone()
    
    def forward(self, adj_matrices: torch.Tensor) -> torch.Tensor:
        """Compute qubit embeddings for a (potentially padded) circuit.

        Args:
            adj_matrices: [S, Q, Q] adjacency tensor. Q can be <= max_qubits.
        """
        num_slices = adj_matrices.shape[0]
        num_qubits = adj_matrices.shape[1]

        if num_qubits > self.max_qubits:
            raise ValueError(
                f"adjacency has {num_qubits} qubits but max_qubits={self.max_qubits}."
            )
        
        if self.use_learnable_ids:
            ids = self.qubit_ids.weight[:num_qubits]  # [Q, D]
        else:
            # Use cached orthogonal IDs
            ids = self.qubit_ids[:num_qubits]  # [Q, D]

        # Flatten to node list
        nodes = ids.unsqueeze(0).expand(num_slices, -1, -1).reshape(
            num_slices * num_qubits, -1
        )

        # Only when we use the temporal transformer do we need to compute GNN embeddings in this step
        # GNN embeddings for lookahead method are computed in that module
        if self.use_temp_transformer:
            # TODO: If we use the temporal transformer + FastTdDataset, we need to convert the dense slices back to sparse here
            # But I'm leaving it as it is for now
            if adj_matrices.layout == torch.sparse_coo:
                edges = adj_matrices._indices()  # [3, num_edges] (batch, slice, src, dst)
            else:
                # Dense tensors: extract the non-zero connections as edges
                edges = torch.nonzero(adj_matrices, as_tuple=False).t()  # [3, num_edges]

            edges = edges.to(nodes.device)
            if edges.numel() == 0 or edges.size(1) == 0:
                out = nodes  # No edges -> fall back to base embeddings
            else:
                src = edges[0] * num_qubits + edges[1]
                dst = edges[0] * num_qubits + edges[2]
                edge_index = torch.stack([src, dst], dim=0)

                # GNN forward
                out = self.slice_encoding(nodes, edge_index)  # [B*S*Q, D]
            out = out.view(num_slices, num_qubits, -1)
            # Positional encoding (per slice)
            if self.positional_encoding is not None:
                out = self.positional_encoding(out)
            return out
        
        return nodes.view(num_slices, num_qubits, -1) # [num_slices, num_qubits, hidden_dim] #TODO: Not ideal
    


class QubitContextTransformer(nn.Module):
    def __init__(self, hidden_dim, num_layers=2, num_heads=4, dropout=0.1, normalization="instance", use_reverse_causal=False):
        super().__init__()
        self.use_reverse_causal = use_reverse_causal
        # Note: an error is thrown if we use both masks and causal
        # But here we don't need masks anyway

        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                normalization=normalization,
                bias=True,
                causal=use_reverse_causal,
            ) for _ in range(num_layers)
        ])

    def forward(self, qubit_embeds):
        # qubit_embeds: [num_slices, num_qubits, hidden_dim]
        T, Q, d = qubit_embeds.shape
        x = qubit_embeds.permute(1, 0, 2)  # [num_qubits, num_slices, hidden_dim]
        x = x.reshape(Q, T, d)  
        
        # For reverse causal, flip time dimension
        if self.use_reverse_causal:
            x = torch.flip(x, dims=[1])  
        
        for layer in self.layers:
            x = layer(x, mask=None)
        
        # Flip back if we used reverse causal!
        if self.use_reverse_causal:
            x = torch.flip(x, dims=[1])  
            
        return x.view(Q, T, d).permute(1, 0, 2)  # [num_slices, num_qubits, hidden_dim] 


class QubitContextLookahead(nn.Module):
    """Alternative to QubitContextTransformer using lookahead weights and a GNN.
    """

    def __init__(self, num_qubits, embed_dim, num_layers: int = 2, lookahead_weight: float = 0.5):
        super().__init__()
        self.embed_dim = embed_dim
        # num_qubits acts as an upper bound; actual slices may have fewer qubits
        self.max_qubits = num_qubits
        self.num_layers = num_layers
        self.lookahead_weight = lookahead_weight
        self.init_qubit_embeddings = nn.Embedding(self.max_qubits+1, self.embed_dim, padding_idx=self.max_qubits)
        self.gnn_layers = nn.ModuleList([
            gnn.DenseGCNConv(self.embed_dim, self.embed_dim) for _ in range(self.num_layers)
        ])

    def _get_lookahead_weights(self, slices) -> torch.Tensor:
        """Compute lookahead weights for each slice.

        Accepts either a tensor [B, T, Q, Q] or a dict with key 'slices'.
        """

        if isinstance(slices, dict):
            slices = slices['slices'] # [B, T, Q, Q] sparse tensor

        # We convert to dense for simplicity (advanced indexing is tricky with COO)
        if slices.layout == torch.sparse_coo:
            slices = slices.to_dense().float()
        
        B, T, Q, _ = slices.shape

        weights = torch.zeros_like(slices, dtype=torch.float, device=slices.device)
        # Initialize with last slice
        weights[:, -1] = self.lookahead_weight * slices[:, -1]
        
        # Backward pass through time slices
        for slice_idx in range(T-2, -1, -1):
            weights[:, slice_idx] = self.lookahead_weight * (
                weights[:, slice_idx + 1] + slices[:, slice_idx]
            )

        return weights
    
    def forward(self, agent_embeds, adj_matrices: torch.Tensor) -> torch.Tensor: 
        B = adj_matrices.shape[0]
        T = adj_matrices.shape[1]
        num_qubits = adj_matrices.shape[-1]

        if num_qubits > self.max_qubits:
            raise ValueError(f"adjacency has {num_qubits} qubits but max_qubits={self.max_qubits}")

        weights = self._get_lookahead_weights(adj_matrices)  # [B, S, Q, Q] (dense)
        # Note that the weight matrices will be denser than the slice matrices.
        # So we can pass them directly to the GNN. 

        #base_nodes = self.init_qubit_embeddings.weight[:-1] # [Q, D]
        #base_nodes = agent_embeds
        # Now tile to [B, T, Q, D]
        #nodes = base_nodes.to(device).unsqueeze(0).unsqueeze(0).expand(B, T, self.num_qubits, self.embed_dim)

        x = agent_embeds.view(B * T, num_qubits, self.embed_dim)  # [B*T, Q, D]
        w = weights.view(B * T, num_qubits, num_qubits)  # [B*T, Q, Q]

        for layer in self.gnn_layers:
            x = layer(x, w)

        out = x  # [B*T*Q, D]
        out = out.view(B, T, num_qubits, self.embed_dim) # [B, T, Q, D]

        return out



class QAInitEmbedding(nn.Module):
    """Initial embedding for Qubit Allocation env.

    Produces:
      - Agent (qubit) embeddings (slice-aware)
      - Temporal/global tokens
    """

    def __init__(
        self,
        num_qubits: int = None,
        embed_dim: int = 0,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        normalization: str = "instance",
        use_temp_transformer: bool = True,
        lookahead_weight: float = 0.5,
        use_learnable_qubit_ids: bool = False,
        max_qubits: int = None,
    ):
        super().__init__()
        self.max_qubits = max_qubits if max_qubits is not None else num_qubits
        if self.max_qubits is None:
            raise ValueError("QAInitEmbedding requires max_qubits (or num_qubits as alias).")
        # Agent side
        self.qubit_embds = QubitEmbedding(
            embed_dim, 
            self.max_qubits, 
            dropout=dropout, 
            use_learnable_ids=use_learnable_qubit_ids,
            use_temp_transformer=use_temp_transformer
            )
        if use_temp_transformer:
            self.temporal_qubit_embds = QubitContextTransformer(
                hidden_dim=embed_dim, num_layers=num_layers, num_heads=num_heads, dropout=dropout, 
                normalization=normalization, use_reverse_causal=True
            )

        else:
            self.temporal_qubit_embds = QubitContextLookahead(
                self.max_qubits, embed_dim, num_layers=num_layers, lookahead_weight=lookahead_weight
            )
            self.global_qubit_embds = None

    def forward(self, adj_matrices: torch.Tensor):
        """
        Args:
            adj_matrices: Tensor with shape [S, Q, Q]
            slice_idx: int, index of current slice
        Returns:
            agent_embeds: [Q, d]
            core_embeds:  [C, d]
            slice_token:  [d]
            global_token: [d]
        """
        # Slice GNN embeddings
        agent_gnn_embeds = self.qubit_embds(adj_matrices)  # [S, Q, d]

        # Temporal agent embeddings
        agent_slice_embds = self.temporal_qubit_embds(agent_gnn_embeds)  # [S, Q, d]
        
        return agent_slice_embds
    


class CoreFeatureEncoder(nn.Module):
    """Encode dynamic per-core features into embeddings.

    Args:
        core_size: max capacity per core
        distance_matrix: [num_cores, num_cores] distances
        embed_dim: output embedding dimension
        linear_bias: use bias in linear layers
    """

    def __init__(self,  embed_dim, linear_bias=False):
        super().__init__()

        self.capacity_proj = nn.Linear(1, embed_dim, bias=linear_bias)
        #self.dist_proj = nn.Linear(distance_matrix.size(-1), embed_dim, bias=linear_bias)

    def forward(self, core_capacities: torch.Tensor, core_size: torch.tensor):
        """
        Args:
            td: TensorDict containing
                - 'current_core_capacity': [C]
                - 'last_assignment': [Q]
        Returns:
            core_feats: [C, d]
        """
        # capacities are given without the batch size
        # core_capacities has shape [C+1] where the last entry is the buffer
        # we remove it here. 
        cap = core_capacities[:-1].unsqueeze(-1) / core_size.unsqueeze(-1)  # normalize [0,1]
        cap_emb = self.capacity_proj(cap)  # [C,d]

        # distance-based features
        # Here, we just use static per-core distances as embedding
        #dist_emb = self.dist_proj(self.distance_matrix.unsqueeze(0).expand(td.batch_size[0], -1, -1))

        return cap_emb.unsqueeze(0)  # [C, d]
    


class AgentBinder(nn.Module):
    """
    Performs binding  of qubit identity and interaction costs.

    Motivation:
        In a pair-agent setup, simply pooling qubit embeddings and distance embeddings separately
        introduces a commutativity ambiguity. The model sees the total pair features and the 
        total distance cost, but loses the specific attribution of which qubit pays which cost.
        
        Example: 
        - Case A: hub qubit (high value) at dist 0 + leaf qubit (low value) at dist 2.
        - Case B: hub qubit at dist 2 + leaf qubit at dist 0.
        
        Both sum to the same total distance (2), but case A is better strategically.

    Mechanism:
        This module fuses the individual qubit embedding [d] with its specific distance embedding [d]
        via a non-linear MLP before any pooling occurs. 
    """

    def __init__(self, embed_dim):
        super().__init__()
        # 2*d -> d for efficiency (i will try try larger later)
        self.proj_in = nn.Linear(2 * embed_dim, embed_dim)
        self.act = nn.GELU() # or nn.ReLU(), TODO: ablate
        self.proj_out = nn.Linear(embed_dim, embed_dim)
        # add residual so the model still knows who the qubit is
        self.resid_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, qubit_emb, dist_emb):
        """
        qubit_emb: [Q, C, d] (Expanded)
        dist_emb:  [Q, C, d] (Projected)
        """
        # [Q, C, 2d]
        combined = torch.cat([qubit_emb, dist_emb], dim=-1)
        
        # [Q, C, d]
        fused = self.proj_in(combined)
        fused = self.act(fused)
        fused = self.proj_out(fused)

        # TODO: Ablate this residual scale
        return fused + (self.resid_scale * qubit_emb)


    
class DynamicAgentGrouper(nn.Module):
    """
    Computes distances for all qubits to all cores.
    Binds qubit identity + distance using AgentBinder.
    Groups qubits into pairs and singletons.
    Returns a padded tensor of agents
        """
    def __init__(self, embed_dim: int):
        super().__init__()

        # project scalar distance to embedding vector
        self.dist_proj = nn.Linear(1, embed_dim, bias=False)

        self.binder = AgentBinder(embed_dim)

        # Projection after concatenating member embeddings
        self.concat_proj = nn.Linear(embed_dim * 2, embed_dim)

        self.q_to_agent = None

    
    def _get_dist(self, 
        prev_core_allocs, 
        core_connectivity,
        current_core_allocs,
        ):
        """
        Computes [Q, C] distance matrix from the current core allocations (qubits allocated in current
        time slice) and previous core allocations (qubits yet to be allocated)
        """
        num_cores = core_connectivity.size(0)
        is_buffer_prev = prev_core_allocs >= num_cores # [Q]
        is_buffer_curr = current_core_allocs >= num_cores # [Q] 
        # for allocated qubits, we take the current allocation as the previous allocation for distance calculation
        # this gives information on the current state of the allocation
        prev_core_allocs[~is_buffer_curr] = current_core_allocs[~is_buffer_curr] 
        safe = prev_core_allocs.clamp(0, num_cores - 1) # [Q]
        dist = core_connectivity.index_select(0, safe.long()) # [Q, C]
        # if qubit is in the buffer, replace its row of distances with zeros
        is_buffer = is_buffer_prev | is_buffer_curr
        dist = torch.where(is_buffer.unsqueeze(-1), dist.new_zeros(1), dist)
        return dist  # [Q, C]
    
    
    def _extract_agent_indices(
        self,
        adj_matrix: torch.Tensor,
        current_core_allocs: torch.Tensor,
        num_cores: int,
        device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute pair (q1,q2) and singleton indices given adjacency and allocations.
        NOTE: We only consider unassigned agents!!!
        """
        unassigned = current_core_allocs == num_cores  # only buffer qubits can be allocated this step

        pair_mask = torch.triu(adj_matrix, 1) > 0
        q1, q2 = torch.where(pair_mask)
        keep_pairs = unassigned[q1] & unassigned[q2]
        q1, q2 = q1[keep_pairs], q2[keep_pairs]

        single_used = torch.zeros(current_core_allocs.size(0), dtype=torch.bool, device=device)
        single_used[q1] = True
        single_used[q2] = True
        q_single = torch.where(unassigned & ~single_used)[0]
        return q1, q2, q_single
    

    def build_mapping(self, Q: int, q1: torch.Tensor, q2: torch.Tensor, q_single: torch.Tensor, device) -> torch.Tensor:
        """
        Build a qubit -> agent mapping 
        """
        q_to_agent = torch.full((Q,), 0, dtype=torch.long, device=device)
        pair_count = q1.numel()
        if pair_count:
            pair_pos = torch.arange(pair_count, device=device)
            q_to_agent[q1] = pair_pos
            q_to_agent[q2] = pair_pos
        if q_single.numel():
            single_pos = torch.arange(q_single.numel(), device=device) + pair_count
            q_to_agent[q_single] = single_pos
        return q_to_agent
    
    
    def agents_to_qubits(
        self, tensor: torch.Tensor, 
        q_to_agent: Optional[torch.Tensor] = None, 
        current_core_allocs: Optional[torch.Tensor] = None, 
        num_cores: int = None) -> torch.Tensor:
        """
        Broadcast agent-dim tensor back to qubit dim using q_to_agent.
        If current_core_allocs is provided, it restores the value of already allocated 
        qubits instead of taking the value from the mapped agent (which defaults to agent 0).
        """
        if q_to_agent is None:
            if self.q_to_agent is None:
                raise ValueError("q_to_agent not provided and no cached mapping available")
            q_to_agent = self.q_to_agent
        
        a_dim = tensor.ndim - 1 # Agent dim 
        
        view_shape = [1] * a_dim + [q_to_agent.numel()]
        idx = q_to_agent.view(view_shape)
        
        final_shape = list(tensor.shape)
        final_shape[a_dim] = q_to_agent.numel()
        
        idx = idx.expand(final_shape)
        out = torch.gather(tensor, a_dim, idx)
        
        if current_core_allocs is not None:
             if num_cores is None:
                 raise ValueError("num_cores must be provided if current_core_allocs is provided")
             is_active_qubit = (current_core_allocs == num_cores)
             out = torch.where(is_active_qubit, out, current_core_allocs)
        
        return out
    

    def qubits_to_agents(
        self,
        qubit_embeds: torch.Tensor,# [Q, D]
        adj_matrix: torch.Tensor, # [Q, Q]
        prev_core_allocs: torch.Tensor, # [Q]
        current_core_allocs: torch.Tensor, # [Q]
        core_connectivity: torch.Tensor,  # [C, C]
        action_mask: Optional[torch.Tensor] = None,  # [Q, C+1]
    ):
        Q, _ = qubit_embeds.shape
        C = core_connectivity.size(0)
        device = qubit_embeds.device

        # Distances and binding
        dist = self._get_dist(prev_core_allocs, core_connectivity, current_core_allocs) # [Q, C]
        dist_emb = self.dist_proj(dist.unsqueeze(-1)) # [Q, C, d]
        qubit_expanded = qubit_embeds[:, None, :].expand(-1, dist_emb.size(1), -1) # [Q, C, d]
        bound = self.binder(qubit_expanded, dist_emb) # [Q, C, d]

        mask = action_mask if action_mask is not None else torch.ones(Q, C + 1, dtype=torch.bool, device=device)
        if mask.dim() != 2:
            raise ValueError("action_mask must be [Q, C+1]")

        # Pairs and singles
        q1, q2, q_single = self._extract_agent_indices(adj_matrix, current_core_allocs, C, device)

        # TODO: Ablate: now we're just summing the embds of two qubits in a pair
        pair_agents = bound[q1] + bound[q2] # [P, C, d]
        single_agents = bound[q_single]# [S, C, d]

        agents = torch.cat([pair_agents, single_agents], dim=0)
        demands = torch.cat([torch.full((pair_agents.size(0),), 2.0, device=device),
                            torch.full((single_agents.size(0),), 1.0, device=device)])
        action_masks = torch.cat([mask[q1], mask[q_single]], dim=0)

        q_to_agent = self.build_mapping(Q, q1, q2, q_single, device)
        self.q_to_agent = q_to_agent  # cache mapping

        return agents, demands > 0, demands, action_masks, agents.size(0)


    def forward(
        self,
        qubit_embeds: torch.Tensor,
        adj_matrix: Optional[torch.Tensor] = None,
        prev_core_allocs: Optional[torch.Tensor] = None,
        current_core_allocs: Optional[torch.Tensor] = None,
        core_connectivity: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,

    ):
        agent_embeds, agent_mask, agent_demands, final_action_mask, _ = self.qubits_to_agents(
            qubit_embeds,
            adj_matrix=adj_matrix,
            prev_core_allocs=prev_core_allocs,
            current_core_allocs=current_core_allocs,
            core_connectivity=core_connectivity,
            action_mask=action_mask
        )

        return agent_embeds, agent_mask, agent_demands, final_action_mask
    

