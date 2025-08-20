import io
from typing import Any, Dict, Optional
import gymnasium as gym
import numpy as np
from fgproee.alg.fgp import buildLookaheadWeights
from sampler.randomcircuit import RandomCircuit


class QubitAllocationEnv(gym.Env):
    """ Qubit Allocation Environment (gym.Env subclass) described in Ref. [1].

    Observation (Dict):
      - node_features: (N_max, 2*C + 1 [+1 if use_extended])
          [0:C) -> one-hot past allocation per qubit
          [C:2C) -> one-hot current allocation per qubit
          [2C] -> target-qubit flag (1 if this is q_f)
          [2C+1]* -> optional interacting-qubit flag for q_f 
      - A1: (N_max, N_max) current-slice interactions (0/1)
      - A2: (N_max, N_max) lookahead interactions (0/inf)
      - Z: (C,) remaining capacity per core (float32)
      - R: (C,) reservation target-qubit index per core (int32, padded with -1)
      - N: () number of qubits in current circuit (int32)

    Action: Discrete(C) -> choose a core index for current qubit.

    Reward (negative float):
    reward =  - alpha * nonlocal_comm(a,s) - beta  * intervention(a,s) - gamma * direct_capacity_violation(a,s)

    Notes:
      - N_max and C are fixed at construction time.
      - Circuits of N<=N_max are loaded at reset.

    References:
    [Leveraging Graph-Based Reinforcement Learning for Qubit Allocation on Multi-Core Quantum Computing Architectures].
    Sergi Tomás Martínez, 2025.
    """
    
    def __init__(self, 
                 N_max: int = 50,
                 C: int = 5,
                 use_extended: bool = False,
                 alpha: float = 0.1,
                 beta: float = 0.1,
                 gamma: float = 0.9,
                 initial_capacity: int = 2,
                 seed: Optional[int] = None):
        super().__init__()

        self.rng = np.random.default_rng(seed)

        self.N_max = int(N_max)
        self.C = int(C)
        self.use_extended = bool(use_extended)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.initial_capacity = int(initial_capacity)

        nf_cols = 2 * self.C + 1 + (1 if self.use_extended else 0) 

        self.action_space = gym.spaces.Discrete(self.C)
        self.observation_space = gym.spaces.Dict({
            "node_features": gym.spaces.Box(low=0.0, high=1.0, shape=(self.N_max, nf_cols), dtype=np.float32),
            "A1": gym.spaces.Box(low=0.0, high=1.0, shape=(self.N_max, self.N_max), dtype=np.float32),
            "A2": gym.spaces.Box(low=0.0, high=np.finfo(np.float32).max, shape=(self.N_max, self.N_max), dtype=np.float32),
            "Z": gym.spaces.Box(low=0.0, high=np.finfo(np.float32).max, shape=(self.C,), dtype=np.float32), # Core capacities
            "R": gym.spaces.Box(low=-1, high=self.C -1, shape=(self.N_max,), dtype=np.int32), # Reservations (eq. 21)
            "N": gym.spaces.Box(low=0, high=self.N_max, shape=(), dtype=np.int32), # Actual number of qubits
        })

        self.N: int = 0 
        self.T: int = 0
        self.current_q: int = 0  
        self.current_t: int = 0

        self.A1 = np.zeros((self.N_max, self.N_max), dtype=np.float32)  # Current slice interactions
        self.A2 = np.zeros((self.N_max, self.N_max), dtype=np.float32)  # Lookahead interactions
        self.Z = np.full((self.C,), self.initial_capacity, dtype=np.float32)  # Core capacities
        self.R = np.full((self.N_max,), -1, dtype=np.int32)  # Reservations
        self.node_features = np.zeros((self.N_max, nf_cols), dtype=np.float32)  # Node features

        self.Madj: Optional[np.ndarray] = None
        self.Mweights: Optional[np.ndarray] = None

    def _flag_qubit_idx(self) -> int:
        return 2 * self.C
    
    def _ext_col(self) -> Optional[int]:
        return (2 * self.C + 1) if self.use_extended else None
    
    def _cur_band(self) -> slice:
        return slice(self.C, 2*self.C)
    
    def _past_band(self) -> slice:
        return slice(0, self.C)
    
    def _get_partners(self, q: int):
        """Get indices of qubits that interact with qubit q in the current slice."""
        return np.nonzero(self.A1[q, :self.N] == 1)[0]
    
    def _get_allocation(self, q: int) -> int:
        """Return current allocated core of q, or -1 if unallocated."""
        band = self.node_features[q, self._cur_band()]
        return int(np.argmax(band)) if band.sum() > 0 else -1

    def _allocated_mask(self) -> np.ndarray:
        """Boolean mask of which qubits are currently allocated in this slice."""
        return (self.node_features[:, self._cur_band()].sum(axis=1) > 0)

    ## VIOLATION MASKS

    def _direct_capacity_violation_mask(self, mask=None):
        """Mask out cores that are at full capacity."""
        if mask is None:
            mask = np.ones(self.C, dtype=bool)

        return mask & (self.Z > 0)

    def _reservation_violation_mask(self, mask=None):
        """Reservation capacity: if Z==1 and there exists another pending qubit reserved to 'a'
        Compute a boolean per core with exactly one slot AND has pending reservations."""
        if mask is None:
            mask = np.ones(self.C, dtype=bool)

        return mask & ~((self.Z == 1) & (self._cache_pending_reservations > 0))

    def _interaction_violation_mask(self, mask=None):
        """Interaction: if any friend is already allocated, qf must go to one of those cores."""

        if mask is None:
            mask = np.ones(self.C, dtype=bool)

        if self._cache_partner_cores.size:
            allowed = np.isin(np.arange(self.C), self._cache_partner_cores)
            mask &= allowed

        return mask
    
    def interaction_missing_space_mask(self, mask=None):
        """Missing space for interaction: if any unallocated partner exists, need Z >= 2"""

        if mask is None:
            mask = np.ones(self.C, dtype=bool)  

        if self._cache_has_unallocated_partners:
            mask &= (self.Z >=2)

        return mask

    def _check_future_interaction_missing_space(self, mask=None):
        """If any future interacting pair exists, then after placing qf on 'a' there must still exist a core with >= 2 slots.
        This is equivalent to: max(Z - e_a) >= 2, where e_a is one-hot for core 'a'
        For more details, see Figure 11 in Ref. [1].
        """

        if mask is None:
            mask = np.ones(self.C, dtype=bool)

        if self._cache_any_future_pair:
            max1 = self.Z.max()
            cnt_max1 = (self.Z == max1).sum() # count of cores with max capacity
            if self.C >= 2:
                max2 = np.partition(self.Z, -2)[-2] # efficiently gets the second largest value
            else:
                max2 = -np.inf

            # For each a, the new maximum after subtracting 1 from Z[a]:
            # if Z[a] == max1 and it's unique, new_max = max(max1 - 1, max2)
            # if Z[a] == max1 and it's not unique, new_max = max1 (since others keep max1)
            # if Z[a] <  max1, new_max = max1
            is_max1 = (self.Z == max1)
            new_max_if_place = np.where(
                is_max1 & (cnt_max1 == 1),
                np.maximum(max1 - 1, max2),
                max1
            )
            mask &= (new_max_if_place >= 2)

        return mask

    def _nonlocal_comm(self, action):
        """Calculate the non-local communication cost for the action in the given slice."""
        past = self.node_features[self.current_q, self._past_band()]
        if self.current_t == 0:
            return 0.0  
        past_core = int(np.argmax(past))
        return float(past_core != action)
    

    def _refresh_caches(self):
        """Refresh caches for the current qubit and slice. Caches are used to speed up the action masking."""
        N, C, qf = self.N, self.C, self.current_q

        # Get current allocations. If unallocated, return -1
        cur_band = self._cur_band() 
        cur = self.node_features[:N, cur_band] # (N, C)
        alloc_mask = cur.sum(axis=1) > 0 # sums all values in row N, should be 1 at most (N,)
        cur_alloc = np.where(alloc_mask, np.argmax(cur, axis=1), -1)  # (N,)

        partners = self._get_partners(qf) # Indices of friends of qf in the current slice
        # Get the current allocations of qf's friends and cores
        p_alloc_mask = alloc_mask[partners]  # (P,)
        p_alloc_cores = cur_alloc[partners]  # (P,)
        self._cache_partner_cores = p_alloc_cores[p_alloc_mask]  # (P',) where P' is the number of allocated friends
        
        p_unallocated_mask = ~p_alloc_mask  # unallocated friends
        self._cache_has_unallocated_partners = bool(p_unallocated_mask.any())  # (P'',) where P'' is the number of unallocated friends

        # Pending allocated qubits after qf in the current slice
        idx = np.arange(N)
        pending_mask = (~alloc_mask) & (idx > qf)  # (N,)
        pending_ids = idx[pending_mask]  # (P''',) where P''' is the number of pending qubits

        # Count how many pending qubits are reserved to each core
        if pending_ids.size > 0:
            rcores = self.R[pending_ids]  # (P''',) 
            rcores = rcores[rcores >= 0]  # Filter out -1 (unreserved)
            self._cache_pending_reservations = np.bincount(rcores, minlength=self.C)  # (C,) count of pending reservations per core
        else:
            self._cache_pending_reservations = np.zeros(self.C, dtype=np.int32)

        # Check if there is any future interacting pair among the pending qubits
        if pending_ids.size >= 2:
            # Get the future interactions of the pending qubits
            subA = self.A1[np.ix_(pending_ids, pending_ids)]  # note: extracts a submatrix of A1 for pending qubits (P''', P''')
            self._cache_any_future_pair = bool((subA == 1).any()) # (P''', P''')
        else:
            self._cache_any_future_pair = False

        self._cache_alloc_mask = alloc_mask
        self._cache_cur_alloc = cur_alloc


    def valid_action_mask(self):

        mask = np.ones(self.C, dtype=bool)  

        mask = self._direct_capacity_violation_mask(mask) 
        mask = self._reservation_violation_mask(mask)
        mask = self._interaction_violation_mask(mask)  
        mask = self.interaction_missing_space_mask(mask)  
        mask = self._check_future_interaction_missing_space(mask)  

        return mask

    
    def _get_obs(self):
        return {
            "node_features": self.node_features.copy(),
            "A1": self.A1.copy(),
            "A2": self.A2.copy(),
            "Z": self.Z.copy(),
            "R": self.R.copy(),
            "N": np.array(self.N, dtype=np.int32)
        }

    def step(self, action):
        # action: integer in [0, C-1], selects core for current qubit
        a = int(action)
        q = self.current_q
        assert self.action_space.contains(a), f"Invalid action {a}"

        intervened = False
        
        self._refresh_caches()
        valid_mask = self.valid_action_mask()
        valid_indices = np.flatnonzero(valid_mask)

        attempted_direct_capacity_violation = not self._direct_capacity_violation_mask()[a]

        if a not in valid_indices:
            intervened = True
            if valid_indices.size > 0:
                effective_action = int(self.rng.choice(valid_indices))
            else:
                # Shouldn't happen, but if no valid actions, choose randomly and raise a warning
                with_space = np.flatnonzero(self.Z > 0)
                effective_action = int(self.rng.choice(with_space)) if with_space.size > 0 else int(self.rng.integers(0, self.C))
                raise Warning(f"No valid actions for qubit {q}, using random action {effective_action}.")
        else:
            effective_action = a

        
        # Compute the reward 
        nonlocal_comm = self._nonlocal_comm(effective_action)
        reward = - self.alpha * nonlocal_comm - self.beta * intervened - self.gamma * attempted_direct_capacity_violation

        # Transition function 
        self.node_features[q, self._cur_band()] = 0.0
        self.node_features[q, self._cur_band().start + effective_action] = 1.0
        self.node_features[q, self._flag_qubit_idx()] = 0.0
        self.Z[effective_action] -= 1  # Decrease capacity of the selected core

        partners = self._get_partners(q)  # Friends of the current qubit
        future_partners = partners[partners > q]  # Friends that are after the current qubit
        self.R[future_partners] = effective_action  # Reserve the core for future interactions

        if q + 1 < self.N:
            self.current_q += 1
            self.node_features[self.current_q, self._flag_qubit_idx()] = 1.0  # Flag the next qubit
            extended_col = self._ext_col()
            if extended_col is not None:
                self.node_features[:, extended_col] = 0.0
                next_partners = self._get_partners(self.current_q)
                self.node_features[next_partners, extended_col] = 1.0  # next qubit's partners

            terminated = False
        else:
            self.node_features[:, self._past_band()] = self.node_features[:, self._cur_band()]
            self.node_features[:, self._cur_band()] = 0.0
            self.Z[:] = float(self.initial_capacity)
            self.R[:self.N] = -1

            self.current_t += 1
            self.A1.fill(0.0)
            self.A2.fill(0.0)
            if self.Madj is not None and self.current_t < self.T:
                n = self.N
                self.A1[:n, :n] = self.Madj[self.current_t, :n, :n]
            if self.Mweights is not None and self.current_t < self.T:
                n = self.N
                self.A2[:n, :n] = self.Mweights[self.current_t, :n, :n]

            terminated = (self.current_t >= self.T)
            self.current_q = 0
            self.node_features[:, self._flag_qubit_idx()] = 0.0
            if not terminated:
                self.node_features[self.current_q, self._flag_qubit_idx()] = 1.0
                ec = self._ext_col()
                if ec is not None:
                    self.node_features[:, ec] = 0.0
                    nxt_partners = self._get_partners(self.current_q)
                    self.node_features[nxt_partners, ec] = 1.0
        
        truncated = False # Shouldn't be used since we have interventions

        observation = self._get_obs()
        info = {
            "attempted_action": a,
            "effective_action": effective_action,
            "intervened": intervened,
            "attempted_direct_capacity_violation": attempted_direct_capacity_violation,
            "current_t": int(self.current_t),
            "current_q": int(self.current_q),
        }

        return observation, reward, terminated, truncated, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        options = options or {}
        self.N = int(options.get("N", self.N_max))
        self.N = max(1, min(self.N, self.N_max))
        self.slice_index = int(options.get("slice_index", 0))

        # Load or compute A1 and A2 for this episode 
        # pad to N_max if necessary
        Madj = options.get("Madj")  # (N_max, N_max)
        Mweights = options.get("Mweights")  # (N_max, N_max)
        self.T = self.T if Madj is None else options.get("T", Madj.shape[0])
        self.current_t = self.current_t if Madj is None else 0

        assert Madj is None or (Madj.shape[1:] == (self.N, self.N)
                                and Madj.shape[0] == self.T), "Madj must be (T, N, N)"
        assert Mweights is None or (Mweights.shape[1:] == (self.N, self.N)
                                    and Mweights.shape[0] == self.T), "Mweights must be (T, N, N)"
        
        # Store full slice adjency matrices
        self.Madj = Madj.astype(np.float32, copy=False) if Madj is not None else None
        self.Mweights = Mweights.astype(np.float32, copy=False) if Mweights is not None else None

        self.A1.fill(0.0)
        self.A2.fill(0.0)
        if Madj is not None:
            n = min(self.N, self.A1.shape[0])
            self.A1[:n, :n] = Madj[self.current_t, :n, :n].astype(np.float32)
        if Mweights is not None:
            n = min(self.N, self.A2.shape[0])
            self.A2[:n, :n] = Mweights[self.current_t, :n, :n].astype(np.float32)

        # Node features
        self.node_features.fill(0.0)
        # Start with first qubit flagged
        self.current_q = 0
        self.node_features[self.current_q, self._flag_qubit_idx()] = 1.0

        # Capacities and reservations
        self.Z[:] = float(self.initial_capacity)
        self.R[:] = -1

        info: Dict[str, Any] = {}
        return self._get_obs(), info
    
    ### RENDERING METHODS

    def _capacity_bar(self, v: float, max_units: int = 8) -> str:
        if self.initial_capacity <= 0:
            return ""
        units = int(round(max(0.0, min(v, float(self.initial_capacity))) * (max_units / float(self.initial_capacity))))
        return "\u2588" * units + "·" * (max_units - units)

    def _alloc_info(self, q: int) -> str:
        """Show past core, current core, reservation, and a marker for the flagged qubit."""
        past_band = self._past_band()
        cur_band = self._cur_band()
        past = self.node_features[q, past_band]
        cur = self.node_features[q, cur_band]
        past_core = np.argmax(past) if past.sum() > 0 else -1
        cur_core  = np.argmax(cur) if cur.sum()  > 0 else -1
        mark = "\u2192" if q == self.current_q else " "
        return f"{mark} q{q:02d}  past:{past_core:2d}  curr:{cur_core:2d}  R:{self.R[q]:2d}"

    def _edges_in_A1(self) -> list[tuple[int,int]]:
        if self.N <= 1:
            return []
        tri = np.triu(self.A1[:self.N, :self.N], k=1)
        I, J = np.nonzero(tri == 1)
        return list(zip(I.tolist(), J.tolist()))

    def render(self, mode: str = "ansi"):
        if mode != "ansi":
            raise NotImplementedError("Only mode='ansi' is supported.")

        buf = io.StringIO()
        N, C = self.N, self.C

        print("=== QubitAllocationEnv ===", file=buf)
        print(f"slice: t={self.current_t} / T={self.T or 1}    target: q={self.current_q} / N={N}", file=buf)

        self._refresh_caches()
        vmask = self.valid_action_mask()
        allowed = np.flatnonzero(vmask)

        print("\nCapacities Z per core:", file=buf)
        for a in range(C):
            print(f"  core {a}: Z={self.Z[a]:.0f}  {self._capacity_bar(self.Z[a])}", file=buf)

        pending = np.arange(N)[self.node_features[:N, self._cur_band()].sum(axis=1) == 0]
        if pending.size:
            rcores = self.R[pending]
            counts = np.bincount(rcores[rcores >= 0], minlength=C)
        else:
            counts = np.zeros(C, dtype=int)
        print("\nReservations (pending only):", file=buf)
        for a in range(C):
            rlist = [int(q) for q in pending if self.R[q] == a]
            print(f"  core {a}: {counts[a]} -> {rlist}", file=buf)

        print("\nAllocations:", file=buf)
        for q in range(N):
            print("  " + self._alloc_info(q), file=buf)

        edges = self._edges_in_A1()
        print("\nA1 edges (current slice):", file=buf)
        if edges:
            print("  " + ", ".join(f"({i},{j})" for i, j in edges), file=buf)
        else:
            print("  (none)", file=buf)

        print("\nValid actions for current q:", file=buf)
        if allowed.size:
            print(f"  mask: {vmask.astype(int).tolist()}   allowed cores: {allowed.tolist()}", file=buf)
        else:
            print(f"  mask: {vmask.astype(int).tolist()}   allowed cores: (none)  — deadlock", file=buf)


        # print("\nRule diagnostics:", file=buf)
        # print(f"  has_unallocated_partners: {self._cache_has_unallocated_partners}", file=buf)
        # print(f"  any_future_pair: {self._cache_any_future_pair}", file=buf)

        return buf.getvalue()
    
    
class RandomCircuitEnv(gym.Wrapper):
    """A wrapper for the QubitAllocationEnv that provides a random circuit sampler."""
    
    def __init__(self, env, num_lq, num_slices, sampler=None):
        super().__init__(env)
        self.num_lq = num_lq
        self.num_slices = num_slices
        self.sampler = sampler or RandomCircuit(num_lq=num_lq, num_slices=num_slices)

    def reset(self, **kwargs):
        _, A1 = self.sampler.sampleBatch(batch_size=1) # [1, T, N, N]
        A1 = A1.squeeze(0) # [T, N, N]
        #TODO: Clarify the sigma parameter
        A2 = buildLookaheadWeights(A1, sigma=1.0) # [T, N, N]

        options = dict(
            N=self.num_lq,
            Madj=A1.cpu().numpy(),
            Mweights=A2.cpu().numpy(),
            T=self.num_slices
        )
        return self.env.reset(options=options)