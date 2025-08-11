from typing import Dict, List, Tuple, Optional
from fgproee.alg.roee import ROEEPartitioner
import numpy as np
import networkx as nx
import torch
from sampler.randomcircuit import RandomCircuit

NOW_BOOST = 1e12 # Boost for current edges in lookahead weights


def tensorToGraph(adj, weight_name: str = "weight") -> nx.Graph:
    """Builds an undirected weighted NetworkX graph from a (N,N) tensor or numpy array (symmetric, zero diagonal)."""
    if isinstance(adj, torch.Tensor):
        a = adj.detach().cpu().numpy().astype(np.float64, copy=False)
    else:
        a = np.asarray(adj, dtype=np.float64)
    N = a.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(N))
    # Because the adj matrix is symmetric, we can use np.triu_indices to get upper triangle indices
    iu, ju = np.triu_indices(N, k=1) # gives indices of upper triangle (i,j) with i<j, i.e. without the diagonal
    w = a[iu, ju]
    mask = w != 0
    data = [(int(i), int(j), {weight_name: float(w_)}) for i, j, w_ in zip(iu[mask], ju[mask], w[mask])]
    G.add_edges_from(data)
    return G


def buildLookaheadWeights(slice_mats: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Given slice adjacency matrices (T,N,N) with 0/1 edges per slice, produce lookahead-weighted matrices (float):
      W_t = NOW_BOOST * A_t  +  sum_{d=1..T-t-1} 2^{-d/\sigma} * A_{t+d}
    """
    assert slice_mats.dim() == 3 and slice_mats.size(1) == slice_mats.size(2), "slice_mats must be (T,N,N)"
    T, N, _ = slice_mats.shape
    out = torch.zeros_like(slice_mats, dtype=torch.float64)
    if T == 0:
        return out
    # Precompute decays
    if T > 1:
        dists = torch.arange(1, T, dtype=torch.float64)
        decays = torch.pow(torch.tensor(2.0, dtype=torch.float64), -dists / float(sigma))
    for t in range(T):
        # Set weights of the current slice to a very large number
        out[t] += slice_mats[t].to(torch.float64) * NOW_BOOST
        for d in range(1, T - t):
            out[t] += slice_mats[t + d].to(torch.float64) * decays[d - 1]
        # Keep diagonal zero
        out[t].fill_diagonal_(0.0)
    return out



class FineGrainedPartitionerROEE:
    """
    Orchestrate the time-sliced rOEE with lookahead weights (exponential decay).
    - For each slice t, build G_t^LA with NOW_BOOST for current edges and exponentially-decayed weights for future edges.
    - Run rOEE on G_t^LA seeded with previous assignment.
    - Return per-slice assignments and costs (on the lookahead-weighted graph of that slice).
    """
    def __init__(self, k: int, sigma: float = 1.0, max_oee_passes: int = 100,seed: Optional[int] = None):
        self.k = k
        self.sigma = float(sigma)
        self.seed = seed
        self.max_oee_passes = max_oee_passes

    def run(self, slice_mats: torch.Tensor, verbose: bool = False):
        """
        slice_mats: (T, N, N) 0/1 adjacency per slice.
        Returns:
            maps: List[Dict[node->cluster]] length T
            costs: List[float] per-slice cut costs on lookahead-weighted graphs
        """
        T, N, _ = slice_mats.shape
        # Build lookahead-weighted slice graphs
        W = buildLookaheadWeights(slice_mats, sigma=self.sigma)  # (T,N,N)
        # For validity check we need the original current-slice adjacency, not lookahead weights
        maps: List[Dict[int,int]] = []
        costs: List[float] = []
        init_map: Optional[Dict[int,int]] = None
        for t in range(T):
            if verbose:
                print(f"== Slice {t}/{T-1} ==")
            Gt = tensorToGraph(W[t], "weight") # Guide rOEE with lookahead weights
            if init_map is None:
                part = ROEEPartitioner(Gt, k=self.k, init=None, seed=self.seed)
            else:
                part = ROEEPartitioner(Gt, k=self.k, init=init_map, seed=None)
            sl_adj = slice_mats[t].detach().cpu().numpy().astype(np.float64, copy=False)
            part_map, cost, p_used = part.run(sl_adj, verbose=verbose, max_passes=self.max_oee_passes)
            if verbose:
                print(f"   passes used: {p_used}, slice cost: {cost:.3f}")
            maps.append(part_map)
            costs.append(cost)
            init_map = part_map  # seed next slice
        return maps, costs