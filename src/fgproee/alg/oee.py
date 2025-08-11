from typing import Dict, List, Tuple, Optional
import numpy as np
import networkx as nx
import random


class OEECore:
    """Balanced k-way OEE algorithm on a weighted undirected graph. 
    This is a core implementation that can be used to build OEE-based partitioners. 
    Based on the original OEE algorithm, but adapted to work with weighted graphs.
    Args:
        A (np.ndarray): Adjacency matrix of the graph (N,N) with weights.
        k (int): Number of clusters (cores) to partition the graph into.
    Returns:
        An instance of OEECore that can be used to run OEE passes and compute costs.
    Raises:
        ValueError: If the number of nodes is not divisible by k (balanced partitioning).
    
    """
    def __init__(self, A: np.ndarray, k: int):
        self.A = A.astype(np.float64, copy=False)
        self.n = A.shape[0]
        self.k = k
        if self.n % k != 0:
            raise ValueError(f"Balanced OEE requires n % k == 0 (n={self.n}, k={k})")
        self.m = self.n // k

    @staticmethod
    def currentCutCost(A: np.ndarray, color: np.ndarray) -> float:
        iu, ju = np.triu_indices(A.shape[0], 1) 
        cross = color[iu] != color[ju] # which pairs are in different clusters
        return float(A[iu, ju][cross].sum())

    @staticmethod
    def initialPartition(A: np.ndarray, k: int, seed: Optional[int] = None) -> np.ndarray:
        """Greedy balanced initialization: favors within-cluster connectivity."""
        rng = random.Random(seed)
        n = A.shape[0]
        m = n // k
        nodes = list(range(n))
        rng.shuffle(nodes)
        color = -np.ones(n, dtype=np.int32)
        cluster_nodes = [set() for _ in range(k)]
        # Seed clusters - this is just to initialize the first k nodes
        for c in range(k):
            u = nodes.pop()
            cluster_nodes[c].add(u)
            color[u] = c
        # Assign rest
        for u in nodes:
            # If the core is empty, the score is 0
            # The score for placing an unassigned qubit u into core c is total edge weight from u to the qubits already in c
            scores = np.array([A[u, list(cluster_nodes[c])].sum() if cluster_nodes[c] else 0.0 for c in range(k)])
            sizes = np.array([len(cluster_nodes[c]) for c in range(k)]) # Number of qubits in each core
            feasible = sizes < m
            scores[~feasible] = -np.inf # Full cores are infeasible
            cands = np.flatnonzero(scores == np.max(scores)) # Give all candidates with max score 
            c = rng.choice(cands.tolist())
            cluster_nodes[c].add(u)
            color[u] = c
        return color
    

    @staticmethod
    def computeW(A: np.ndarray, color: np.ndarray, k: int) -> np.ndarray:
        """
        Total edge weight from qubit i to all qubits currently in core l:
        W[i,l] = sum_{j in C_l} A[i,j]
        We compute it as:
        W = A @ M, where M[:, l] = 1 if node in cluster l, else 0.
        """
        n = color.shape[0]
        M = np.zeros((n, k), dtype=np.float64)
        M[np.arange(n), color] = 1.0  # Compute membership matrix
        W = A @ M  # (n, k)
        return W

    @staticmethod
    def computeD(W: np.ndarray, color: np.ndarray) -> np.ndarray:
        """
        Relative benefit matrix D, where D[i,l] = W[i,l] - W[i,color[i]] (qubit i, core l).
        """
        base = W[np.arange(W.shape[0]), color]
        return W - base[:, None]


    def passBuild(self, color: np.ndarray) -> Tuple[List[Tuple[int,int,int,int]], List[float]]:
        """
        Build one OEE pass: returns list of chosen (a,b,Acl,Bcl) and their gains in order. 
        This means swap qubit a in core Acl with qubit b in core Bcl.
        Within a single pass, each qubit is moved at most once. 
        Gains calculater later in the pass must reflect earlier hypothetical moves, so we keep a temporary evolving assignment in `temp_color`.
        After this returns, the caller will compute the best prefix of these swaps to actually commit.
        """
        A = self.A
        k = self.k
        n = self.n

        temp_color = color.copy()
        W = self.computeW(A, temp_color, k) 
        D = self.computeD(W, temp_color) 

        C_mask = np.ones(n, dtype=bool) # (n)- Initially all qubits are unlocked
        cluster_masks = [temp_color == l for l in range(k)]  # Gives a mask for each core
        pairs: List[Tuple[int,int,int,int]] = [] # Sequence of chosen swaps (a,b,Acl,Bcl)
        gains: List[float] = [] # Swap gains

        # For efficiency, we use a matrix M to track the current core of each qubit
        M = np.zeros((n, k), dtype=np.float64) # (n,k)
        M[np.arange(n), temp_color] = 1.0

        for _ in range(n // 2): # We visit core pairs until all qubits are locked
            best_gain = -np.inf
            best = None

            for Acl in range(k):
                Ia = np.flatnonzero(cluster_masks[Acl] & C_mask) # Indices of qubits in core Acl that are still unlocked (candidates)
                if Ia.size == 0:
                    continue
                Da = D[Ia, :]  # (|Ia|, k)
                for Bcl in range(k): # Iterate over destination cores
                    if Bcl == Acl:
                        continue
                    Jb = np.flatnonzero(cluster_masks[Bcl] & C_mask) # Candidates in core Bcl
                    if Jb.size == 0:
                        continue
                    DiB = Da[:, Bcl][:, None] # (|Ia|,1)
                    DjA = D[Jb, Acl][None, :] # (1,|Jb|)
                    # Gain matrix for all pairs (a in Ia, b in Jb):
                    # g(a,b) = D[a,Bcl] + D[b,Acl] - 2 * A[a,b]
                    Gmat = DiB + DjA - 2.0 * A[np.ix_(Ia, Jb)]
                    idx = np.argmax(Gmat)
                    gi = Gmat.flat[idx]  # Maximal gain within this pair of cores
                    if gi > best_gain:
                        ra, cb = np.unravel_index(idx, Gmat.shape)
                        best_gain = float(gi)
                        best = (Ia[ra], Jb[cb], Acl, Bcl)

            if best is None:
                break

            a, b, Acl, Bcl = best
            gains.append(best_gain)
            pairs.append((a, b, Acl, Bcl))

            # Lock the qubits so they aren't chosen again
            C_mask[a] = False
            C_mask[b] = False

            # Apply the temporary swap in temp_color and masks
            cluster_masks[Acl][a] = False
            cluster_masks[Bcl][b] = False
            temp_color[a], temp_color[b] = Bcl, Acl
            cluster_masks[Bcl][a] = True
            cluster_masks[Acl][b] = True

            # Incremental update of W and D for all remaining unlocked nodes
            # Swapping a and b affects only how other qubits connect to Acl and Bcl
            # No other core's membership changes, so we only update W and D for Acl and Bcl
            unlocked = C_mask
            if unlocked.any():
                Ai_a = A[:, a] # Weights to node a
                Ai_b = A[:, b] # Weights to node b
                # a is not in Acl anymore, so remove A(i,a) from W(i,Acl); also add A(i,b) since b is now in Acl
                W[unlocked, Acl] += (Ai_b[unlocked] - Ai_a[unlocked]) 
                W[unlocked, Bcl] += (Ai_a[unlocked] - Ai_b[unlocked])
                base = W[np.flatnonzero(unlocked), temp_color[unlocked]]
                D[unlocked, Acl] = W[unlocked, Acl] - base
                D[unlocked, Bcl] = W[unlocked, Bcl] - base

            # Update M for swapped nodes
            M[a, Acl] = 0.0; M[a, Bcl] = 1.0
            M[b, Bcl] = 0.0; M[b, Acl] = 1.0

            # Recompute W rows for a,b 
            W[[a, b], :] = A[[a, b], :] @ M

            # Update D rows for a,b
            D[a, :] = W[a, :] - W[a, temp_color[a]]
            D[b, :] = W[b, :] - W[b, temp_color[b]]

        return pairs, gains

    def applySwaps(self, color: np.ndarray, swaps: List[Tuple[int,int,int,int]], prefix_len: int) -> np.ndarray:
        """Apply first prefix_len swaps to the color vector.
        This is used after pass_build to commit the best prefix of swaps.
        Note that if we apply swaps in order, out[a] should still equal Acl. This is just a sanity check
        because qubits are locked after being swapped, so we should not have out-of-date labels.
        Args:
            color (np.ndarray): Current assignment, where color[i] = cluster id of node i.
            swaps (List[Tuple[int,int,int,int]]): List of swaps (a,b,Acl,Bcl) to apply.
            prefix_len (int): Number of swaps to apply from the list.
        Returns:
            np.ndarray: Updated color assignment after applying the swaps."""
        
        out = color.copy()
        for i in range(prefix_len):
            a, b, Acl, Bcl = swaps[i]
            # a and b should still be in the expected cores 
            if not (out[a] == Acl and out[b] == Bcl):
                # If out-of-date due to previous swaps, recompute expected labels from out
                Acl, Bcl = out[a], out[b]
            out[a], out[b] = out[b], out[a]
        return out