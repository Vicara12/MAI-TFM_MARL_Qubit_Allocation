import torch


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
    cost += core_con[allocations[i].flatten(), allocations[i+1].flatten()].sum()
  return cost.item()