from typing import TypeAlias, Tuple, Union
import torch
from torch_geometric.utils import dense_to_sparse
from dataclasses import dataclass


GateType: TypeAlias = Tuple[int,int]
CircSliceType: TypeAlias = Tuple[GateType, ...]



@dataclass
class Circuit:
  slice_gates: Tuple[CircSliceType, ...]
  n_qubits: int
  # alloc_steps: refer to the function __getAllocOrder for info on this attribute


  def __post_init__(self):
    self.alloc_steps = self.__getAllocOrder()
    self.n_steps = len(self.alloc_steps)


  def __getAllocOrder(self) -> Tuple[Tuple[int, Union[GateType, Tuple[int]]], ...]:
    ''' Get the allocation order of te qubits for a given circuit.

    Returns a tuple with the allocations to be performed. Each tuple element is another tuple that
    contains the slice the allocation corresponds to and the qubits involved in the allocation, two
    if the qubits belong to a gate in that time slice or a single one if they don't.
    '''
    allocations = []
    for slice_i, slice in enumerate(self.slice_gates):
      free_qubits = set(range(self.n_qubits))
      for gate in slice:
        allocations.append((slice_i, gate))
        free_qubits -= set(gate) # Remove qubits in gates from set of free qubits
      for q in free_qubits:
        allocations.append((slice_i, (q,)))
    return tuple(allocations)
  
  @property
  def n_slices(self) -> int:
    return len(self.slice_gates)



@dataclass
class Hardware:
  core_capacities: torch.Tensor
  core_connectivity: torch.Tensor
  # sparse_core_con: automatically set in init, has the core_connectivity matrix in sparse format
  # sparse_core_ws: weights of the sparse_core_con matrix


  def __post_init__(self):
    ''' Ensures the correctness of the data.
    '''
    assert len(self.core_capacities.shape) == 1, "Core capacities must be a vector"
    assert not torch.is_floating_point(self.core_capacities), "Core capacities must be of dtype int"
    assert all(self.core_capacities > 0), f"All core capacities should be greater than 0"
    assert len(self.core_connectivity.shape) == 2 and \
           self.core_connectivity.shape[0] == self.core_connectivity.shape[1], \
      f"Core connectivity should be a square matrix, found matrix of shape {self.core_capacities.shape}"
    assert torch.all(self.core_connectivity == self.core_connectivity.T), \
      "Core connectivity matrix should be symmetric"
    self.sparse_core_con, self.sparse_core_ws = dense_to_sparse(self.core_connectivity.float())

  
  @property
  def n_cores(self):
    return len(self.core_capacities)
  
  @property
  def n_physical_qubits(self):
    return sum(self.core_capacities).item()