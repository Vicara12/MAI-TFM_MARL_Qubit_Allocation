from typing import Dict, List, Tuple, Optional
from fgproee.alg.oee import OEECore
import numpy as np
import networkx as nx


class ROEEPartitioner:
    """Run rOEE on a single slice: OEE passes but STOP as soon as the slice is 'valid' (all interacting pairs co-located)."""
    def __init__(self, G: nx.Graph, k: int, init: Optional[Dict[int,int]] = None, seed: Optional[int] = None):
        nodes = list(G.nodes())
        self.node_index = {u: i for i, u in enumerate(nodes)}
        self.nodes = nodes
        n = len(nodes)
        A = nx.to_numpy_array(G, nodelist=nodes, weight="weight", dtype=np.float64)
        np.fill_diagonal(A, 0.0)  # Ensure no self-loops
        self.core = OEECore(A, k)
        # Init color
        if init is None:
            self.color = self.core.initialPartition(A, k, seed=seed)
        else:
            # dict or sequence
            if isinstance(init, dict):
                color = np.array([init[u] for u in nodes], dtype=np.int32)
            else:
                color = np.asarray(init, dtype=np.int32)
            # Validate balanced
            counts = np.bincount(color, minlength=k)
            if not np.all(counts == (n // k)):
                raise ValueError("Initial assignment must be balanced across k clusters.")
            self.color = color

    def checkValidity(self, slice_adj: np.ndarray) -> bool:
        """Check that all interacting pairs in the current slice are in the same cluster.
        This condition is used to stop OEE early when the slice is valid, which is called
        relaxed OEE, abbreviated rOEE."""
        iu, ju = np.triu_indices(slice_adj.shape[0], 1) # upper triangle pairs
        mask = slice_adj[iu, ju] > 0 # which pairs are edges in this slice
        if not mask.any():
            return True # If there are no edges, it's trivially valid
        return np.all(self.color[iu[mask]] == self.color[ju[mask]])

        
    def run(self, slice_adj: np.ndarray, max_passes: int = 100, verbose: bool = False) -> Tuple[Dict, float, int]:
        """
        Runs rOEE, the relaxed version of OEE. It makes OEE passes but stops as soon as the assignment is valid for this slice.
        Args:
            slice_adj (np.ndarray): Adjacency matrix of the current slice.
            max_passes (int): Maximum number of passes to run before giving up.
            verbose (bool): If True, print progress and debug information.
        Returns:
            Tuple[Dict, float, int]: A tuple containing:
                - A dictionary mapping nodes to their cluster assignments.
                - The cut cost of the current assignment on the slice adjacency matrix.
                - The number of passes used.
        """
        if self.checkValidity(slice_adj):
            cost = self.core.currentCutCost(self.core.A, self.color)
            return {u: int(self.color[self.node_index[u]]) for u in self.nodes}, cost, 0

        passes = 0
        while passes < max_passes:
            swaps, gains = self.core.passBuild(self.color)
            if not gains:
                # no pair selected, cannot proceed, fallback
                break
            cum = np.cumsum(gains)
            m_best = int(np.argmax(cum) + 1)
            if cum[m_best - 1] <= 0.0:
                # no positive prefix, can't improve 
                break
            # apply best prefix
            self.color = self.core.applySwaps(self.color, swaps, m_best)
            passes += 1
            if verbose:
                print(f"[rOEE] pass {passes}: applied {m_best} swaps (best prefix gain={cum[m_best-1]:.3f})")
            # early stop when slice is valid: rOEE
            if self.checkValidity(slice_adj):
                break

        cost = self.core.currentCutCost(self.core.A, self.color)
        part_map = {u: int(self.color[self.node_index[u]]) for u in self.nodes}
        return part_map, cost, passes
