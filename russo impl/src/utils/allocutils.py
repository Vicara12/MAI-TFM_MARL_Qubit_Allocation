import torch
from typing import Tuple


def cost(allocations: torch.Tensor, core_con: torch.Tensor):
  ''' Compute the cost of the allocation with the given core connectivity matrix.

  The cost is computed as the number of swaps and the cost per swap.

  Args:
    - allocations: matrix of shape [Q,T] where Q = number of logical qubits, T = number of time slices.
    - core_con: matrix of shape [C,C] where C = number of cores. Item (i,j) indicates the cost of
        swapping qubits at cores i and j. No self loops, diagonal must be zero.
  '''
  num_slices = allocations.shape[1]
  cost = 0
  for i in range(num_slices-1):
    cost += core_con[allocations[:,i].flatten(), allocations[:,i+1].flatten()].sum()
  return cost.item()


def validate(allocations: torch.Tensor,
             circuit_slice_gates: Tuple[Tuple[Tuple[int, int], ...], ...],
             core_capacities: Tuple[int, ...]):
  ''' Given an allocation, check wether it is valid.

  Args:
    - allocations: matrix of shape [Q,T] where Q = number of logical qubits, T = number of time slices.
    - circuit_slice_gates: follows the CircuitSampler convention.
    - core_capacities: tuple containing for each core, the number of physical qubits it holds.
  '''
  # Init an array that indicates the core of each physical qubit
  index = 0
  pq_core = [0]*allocations.shape[0]
  for c, c_size in enumerate(core_capacities):
    pq_core[index:index+c_size] = [c]*c_size
  # For all slices, check wether the gates that act on it involve qubits in the same core
  for t in range(allocations.shape[1]):
    slice_allocation = allocations[:,t].squeeze().tolist()
    for gate in circuit_slice_gates[t]:
      pq0 = slice_allocation.index(gate[0])
      pq1 = slice_allocation.index(gate[1])
      if pq_core[pq0] != pq_core[pq1]:
        return False
  return True