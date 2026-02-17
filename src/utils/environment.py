from typing import Optional, Tuple
import torch
from .customtypes import Circuit, Hardware
from .mapping import map_qubit_to_agent, map_agent_to_qubit



class QubitAllocationEnvironment:
  """This is a flexible environment class for qubit allocation.
  It can be reset with a new circuit and hardware"""
  def __init__(self, circuit: Optional[Circuit] = None, hardware: Optional[Hardware] = None,
               validate_solution: bool = False, auto_reset: bool = True):
    self.circuit = circuit
    self.hardware = hardware
    self.validate_solution = validate_solution
    self.allocations = None
    self.current_core_caps = None
    self.current_slice_ = 0
    self.stage = 0
    self.unallocated_qubits = None
    self.current_allocation = None
    self.current_assignment = None
    if auto_reset and circuit is not None and hardware is not None:
      self.reset(circuit, hardware)
  

  def reset(self, circuit: Optional[Circuit] = None, hardware: Optional[Hardware] = None):
    if circuit is not None:
      self.circuit = circuit
    if hardware is not None:
      self.hardware = hardware
    if self.circuit is None or self.hardware is None:
      raise ValueError("Circuit and hardware must be set before resetting the environment")
    device = self.hardware.core_capacities.device
    self.allocations = torch.full(
      size=(self.circuit.n_slices, self.circuit.n_qubits),
      fill_value=self.hardware.n_cores,
      dtype=int,
      device=device,
    )

    self.current_core_caps = torch.empty(
      self.hardware.n_cores + 1,
      dtype=self.hardware.core_capacities.dtype,
      device=device,
    )
    self.current_slice_ = 0
    self.stage = 0
    self._init_current_slice_state()


  def _init_current_slice_state(self) -> None:
    """Reset per-slice tensors so masks/capacities align with the current circuit."""
    device = self.hardware.core_capacities.device
    self.current_core_caps[:-1] = self.hardware.core_capacities
    self.current_core_caps[-1] = self.circuit.n_qubits
    self.unallocated_qubits = torch.ones(
      self.circuit.n_qubits,
      dtype=torch.bool,
      device=device,
    )
    self.current_assignment = torch.full(
      (self.circuit.n_qubits,),
      self.hardware.n_cores,
      dtype=torch.long,
      device=device,
    )
    self.current_allocation = self.allocations[self.current_slice_]
  

  def allocate(self, cores: torch.Tensor) -> int:
    ''' 
    Allocates qubits each timeslice given a vector of actions, one for each qubit.
    This is the multi-agent approach. 
    Contains several asserts to ensure the validity of the solution.
    Also contains masking. 
    '''
    # Capacities are recomputed at each decoding step (successive steps within a slice!)
    # based on the current allocation
    device = self.hardware.core_capacities.device
    cores = cores.to(device)
    core_fill = torch.bincount(cores, minlength=self.hardware.n_cores+1)
    self.current_core_caps[:-1] = self.hardware.core_capacities - core_fill[:-1]
    self.current_core_caps[-1] = self.circuit.n_qubits # TODO: Is this needed? In the agent handler we set this capacity to 
    # a max number. We're doing this twice. 
    
    if self.validate_solution:
      assert self.current_slice_ < self.circuit.n_slices, "Tried to allocate past the end of the circuit"
      assert all(cores >= 0) and all(cores <= self.hardware.n_cores), \
        f"Tried to allocate to core not in [0,{self.hardware.n_cores-1}] or the buffer action {self.hardware.n_cores}"
      assert len(cores) == self.circuit.n_qubits, \
        f"Expected {self.circuit.n_qubits} actions, got {len(cores)}"
      #assert not self.qubit_is_allocated[qubit], f"Tried to allocate qubit {qubit} twice"
      #assert self.hardware.core_capacities[core] > 0, f"Tried to allocate to complete core {core}"
      assert all(self.current_core_caps > 0), \
        f"Capacity violation(s) in core(s) with index(es) {torch.where(self.current_core_caps <= 0)[0]}"

    self.unallocated_qubits = cores == self.hardware.n_cores
    self.current_assignment = cores
    self._advance_stage() # The stage flag determines masking

    # If finished allocation of time slice
    if self.unallocated_qubits.sum().item() == 0:
      if self.validate_solution:
        # Check all gates have their qubits in the same core
        for gate in self.circuit.slice_gates[self.current_slice_]:
          assert self.allocations[self.current_slice_,gate[0]] == self.allocations[self.current_slice_,gate[1]], \
            (f"In time slice {self.current_slice_} allocated qubit {gate[0]} and {gate[0]} to cores "
            f"{self.allocations[self.current_slice_, gate[0]]} and "
            f"{self.allocations[self.current_slice_, gate[1]]}, but they belong to the same gate")
          
      # Store the allocation
      self.allocations[self.current_slice_] = cores
      # Compute the reward
      alloc_cost = self._get_reward(cores)
        
      self.stage = 0
      self.current_slice_ += 1
      if self.current_slice_ < self.circuit.n_slices:
        self._init_current_slice_state()
      else:
        # keep tensors consistent for potential rendering after completion
        self.unallocated_qubits = torch.zeros_like(self.unallocated_qubits)
        self.current_assignment = torch.full_like(cores, self.hardware.n_cores)

      return alloc_cost 
  

  def _advance_stage(self):
    """Multi-stage allocation. First we allocate pair qubits, then the rest. This function checks whether
    all pairs have been allocated. If so, it advances to the next stage. """
    #new_alloc_mask = (self.current_allocation != cores) & (cores != self.hardware.n_cores)  # Mask of newly allocated qubits
    new_alloc_mask = (self.current_assignment != self.hardware.n_cores)  # Mask of allocated qubits in current step 
    pair_q_indices = self.pair_indices.reshape(-1)
    if pair_q_indices.numel() == 0:
      return
    if torch.all(new_alloc_mask[pair_q_indices]):
      self.stage = 1


  def _get_reward(self, cores) -> int:
    """Reward is negative of the cost, so that the agent learns to minimize cost."""
    if self.current_slice_ == 0:
      alloc_cost = 0
    else:
      prev_core = self.allocations[self.current_slice_-1,]
      alloc_cost = self.hardware.core_connectivity[prev_core,cores].sum().item()
    return alloc_cost
  

  def get_mask(self) -> torch.Tensor:
    """Returns a tensor of shape [n_qubits, n_cores+1] with True for valid actions and False for invalid actions. 
    This can be used for action masking in the policy."""
    device = self.hardware.core_capacities.device
    mask = torch.ones((self.circuit.n_qubits, self.hardware.n_cores+1), dtype=torch.bool, device=device)
    pair_q_indices = self.pair_indices.reshape(-1)
    is_pair = torch.zeros(self.circuit.n_qubits, dtype=torch.bool, device=device)
    if pair_q_indices.numel() > 0:
      is_pair[pair_q_indices] = True

    # First, mask out the buffer action for pairs
    mask[is_pair, self.hardware.n_cores] = False

    # Mask out all actions which are not the buffer core in single qubits if stage = 0
    if self.stage == 0:
      mask[~is_pair, :self.hardware.n_cores] = False
    # Mask out buffer action for single qubits if stage = 1
    if self.stage == 1:
      mask[~is_pair, self.hardware.n_cores] = False

    # Mask out all cores that have capacity < 1 for single qubits and < 2 for pair qubits
    mask[:, self.current_core_caps < 1] = False
    pair_cap_mask = self.current_core_caps < 2
    if is_pair.any() and (pair_cap_mask).any():
      mask[is_pair.unsqueeze(1) & pair_cap_mask.unsqueeze(0)] = False

    # Once a qubit is allocated, it cannot be allocated again: mask out all cores for that qubit except their allocation
    # Actively unmask actions that were masked for capacity reasons for allocated qubits
    mask[~self.unallocated_qubits] = False
    mask[~self.unallocated_qubits, self.current_assignment[~self.unallocated_qubits]] = True
    return mask


  def render(self, agent_mapping: Optional[torch.Tensor] = None) -> None:
    """Render a snapshot of the environment using simple ASCII tables.

    Args:
      agent_mapping: Optional qubit -> agent vector produced externally (e.g. DynamicAgentGrouper).
        If omitted, the environment falls back to its own mapping logic when available.
    """
    if self.circuit is None or self.hardware is None:
      print("Environment not initialized. Call reset() first.")
      return

    def _format_table(headers: list[str], rows: list[list[object]]) -> str:
      widths = [len(str(h)) for h in headers]
      for row in rows:
        for idx, cell in enumerate(row):
          widths[idx] = max(widths[idx], len(str(cell)))
      border = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
      def _row_to_line(values):
        cells = [str(val).ljust(widths[idx]) for idx, val in enumerate(values)]
        return "| " + " | ".join(cells) + " |"
      lines = [border, _row_to_line(headers), border]
      for row in rows:
        lines.append(_row_to_line(row))
      lines.append(border)
      return "\n".join(lines)

    def _fmt_core_id(val: int) -> str:
      return "B" if val == self.hardware.n_cores else str(val)

    needs_reset = False
    if self.current_assignment is None:
      self.current_assignment = torch.full(
        (self.circuit.n_qubits,),
        self.hardware.n_cores,
        dtype=torch.long,
        device=self.hardware.core_capacities.device,
      )
      needs_reset = True

    def _normalize_mapping(mapping: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
      if mapping is None:
        return None
      if isinstance(mapping, torch.Tensor):
        return mapping.detach().cpu()
      return torch.as_tensor(mapping, dtype=torch.long)

    external_mapping = _normalize_mapping(agent_mapping)

    try:
      core_caps = self.current_core_caps.detach().cpu().tolist() if self.current_core_caps is not None else None
      header = f"Slice {self.current_slice_}/{self.circuit.n_slices} | Stage {self.stage}"
      sub_header = f"Core caps left: {core_caps}" if core_caps is not None else "Core caps left: <unknown>"
      line_len = max(len(header), len(sub_header))
      print("=" * line_len)
      print(header)
      print(sub_header)
      print(f"Legend: B = buffer/core {self.hardware.n_cores}")
      print("=" * line_len)

      current_alloc_cpu = self.current_assignment.detach().cpu().tolist()
      prev_alloc_cpu = self.prev_slice_allocations.detach().cpu().tolist() if self.current_slice_ > 0 else None

      mask_tensor = self.get_mask().detach().cpu()
      mask_strings: list[str] = []
      for q_idx in range(mask_tensor.shape[0]):
        valid_actions = [
          ("B" if action_idx == self.hardware.n_cores else str(action_idx))
          for action_idx, allowed in enumerate(mask_tensor[q_idx].tolist())
          if allowed
        ]
        mask_strings.append(" ".join(valid_actions) if valid_actions else "-")

      alloc_rows = []
      for q_idx, curr_val in enumerate(current_alloc_cpu):
        prev_val = _fmt_core_id(prev_alloc_cpu[q_idx]) if prev_alloc_cpu is not None else "-"
        alloc_rows.append([q_idx, _fmt_core_id(curr_val), prev_val, mask_strings[q_idx]])
      print("Current vs previous allocation")
      print(_format_table(["Qubit", "Current", "Prev", "Valid actions"], alloc_rows))

      agent_lookup = None
      agent_count = None
      mapping_source = None
      if external_mapping is not None:
        if external_mapping.numel() == self.circuit.n_qubits:
          agent_lookup = external_mapping.tolist()
          agent_count = int(external_mapping.max().item() + 1) if external_mapping.numel() > 0 else 0
          mapping_source = "(from agent grouper)"
        else:
          print("\nProvided agent_mapping has wrong length; ignoring external mapping.")

      if agent_lookup is None and hasattr(self, 'agent_mapping'):
        q_to_agent, num_agents, _ = self.agent_mapping
        agent_lookup = q_to_agent.detach().cpu().tolist()
        agent_count = num_agents
        mapping_source = "(environment default)"

      if agent_lookup is not None:
        print(f"\nTotal agents in slice: {agent_count} {mapping_source}")
      else:
        print("\nAgent mapping unavailable for this render call.")

      unalloc_mask = None
      if self.unallocated_qubits is not None:
        unalloc_mask = self.unallocated_qubits.detach().cpu().tolist()

      def _agent_label(qubit_idx: int) -> str:
        if agent_lookup is None:
          return "-"
        if unalloc_mask is not None and not unalloc_mask[qubit_idx]:
          return "-"
        return str(agent_lookup[qubit_idx])

      pairs = self.pair_indices.detach().cpu()
      if pairs.numel() == 0:
        print("Pairs: none in this slice")
      else:
        pair_rows = []
        for idx, pair in enumerate(pairs.tolist()):
          pair_rows.append([idx, pair[0], pair[1], _agent_label(pair[0])])
        print("Pairs and agent assignment")
        print(_format_table(["#", "qA", "qB", "Agent"], pair_rows))

      if agent_lookup is not None:
        agent_bins: dict[int, list[int]] = {}
        for qubit_id, agent_id in enumerate(agent_lookup):
          if agent_id < 0:
            continue
          if unalloc_mask is not None and not unalloc_mask[qubit_id]:
            continue
          agent_bins.setdefault(agent_id, []).append(qubit_id)
        agent_rows = [[agent_id, " ".join(map(str, qubits))] for agent_id, qubits in sorted(agent_bins.items())]
        print("Agent -> qubits map")
        print(_format_table(["Agent", "Qubits"], agent_rows))

    finally:
      if needs_reset:
        self.current_assignment = None


  @property
  def pair_indices(self) -> torch.Tensor:
    """Returns a tensor of shape [num_pairs, 2] with the indices of the qubits that belong to the same gate in the current slice. 
    This can be used for pairwise allocation."""
    # TODO: This being called every time is too much. Perhaps we should store the pair indices
    gates = self.circuit.slice_gates[self.current_slice_]
    if len(gates) == 0:
      return torch.empty((0, 2), dtype=torch.int64, device=self.current_assignment.device)
    return torch.as_tensor(gates, dtype=torch.int64, device=self.current_assignment.device)
  
  @property
  def current_slice(self) -> int:
    return self.current_slice_

  @property
  def prev_slice_allocations(self) -> torch.Tensor:
    assert self.current_slice_ > 0, "No previous slice"
    return self.allocations[self.current_slice_-1,:].squeeze()
  
  @property
  def finished(self) -> bool:
    return self.current_slice_ == self.circuit.n_slices
  
  @property
  def qubit_allocations(self) -> torch.Tensor:
    assert self.finished, "Tried to get incomplete allocation list"
    return self.allocations
  

ENV_REGISTRY = {
  'qa': QubitAllocationEnvironment,
} 

# ############################# TESTING #############################

# import torch
# from src.sampler.randomcircuit import RandomCircuit


# def make_dummy_circuit(n_qubits: int, n_slices: int):
#   sampler = RandomCircuit(num_lq=n_qubits, num_slices=n_slices)
#   return sampler.sample()

# def make_dummy_hw(n_qubits: int, n_cores: int):
#   # Example: evenly distribute capacity, fully connected except self.
#   cap = torch.tensor([max(1, n_qubits // n_cores)] * n_cores, dtype=torch.int)
#   con = torch.ones((n_cores, n_cores), dtype=torch.float) - torch.eye(n_cores)
#   return Hardware(core_capacities=cap, core_connectivity=con)

# def main():
#   circuit = make_dummy_circuit(n_qubits=6, n_slices=3)
#   hardware = make_dummy_hw(n_qubits=6, n_cores=3)

#   env = MAQubitAllocationEnvironment(circuit, hardware)
#   print("After init:", env.current_slice, env.finished)

#   for t in range(circuit.n_slices):
#     cores = torch.randint(low=0, high=hardware.n_cores, size=(circuit.n_qubits,))
#     cost = env.allocate(cores)
#     print(f"Slice {t} cost={cost} current_slice={env.current_slice} finished={env.finished}")

#   print("Final allocations:", env.qubit_allocations)

#   new_circuit = make_dummy_circuit(n_qubits=4, n_slices=2)
#   new_hw = make_dummy_hw(n_qubits=4, n_cores=2)
#   env.reset(circuit=new_circuit, hardware=new_hw)
#   print("After reset:", env.current_slice, env.finished)

# if __name__ == "__main__":
#   main()