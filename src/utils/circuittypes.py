from typing import TypeAlias, Tuple
import torch
from dataclasses import dataclass


GateType: TypeAlias = Tuple[int,int]
CircSliceType: TypeAlias = Tuple[GateType, ...]



@dataclass
class Circuit:
  slice_gates: Tuple[CircSliceType, ...]
  slice_matrices: torch.Tensor


  def __post_init__(self):
    ''' Ensures the correctness of slice_gates and slice_matrices.
    '''
    assert len(self.slice_gates) == len(self.slice_matrices), \
      (f"Len of slice gates and slice matrices should coincide "
       f"{len(self.slice_gates)} != {len(self.slice_matrices)}")
    assert len(self.slice_matrices.shape) == 3 and  \
           self.slice_matrices.shape[1] == self.slice_matrices.shape[2], \
      f"Slice matrices should be a vector of square matrices, but found shape {self.slice_matrices.shape}"


  @property
  def n_qubits(self) -> int:
    return self.slice_matrices.shape[1]
  
  @property
  def n_slices(self) -> int:
    return len(self.slice_gates)



@dataclass
class Hardware:
  core_capacities: Tuple[int, ...]
  core_connectivity: torch.Tensor


  def __post_init__(self):
    ''' Ensures the correctness of the data.
    '''
    assert all(c > 0 for c in self.core_capacities), f"All core capacities should be greater than 0"
    assert len(self.core_connectivity.shape) == 2 and \
           self.core_connectivity.shape[0] == self.core_connectivity.shape[1], \
      f"Core connectivity should be a square matrix, found matrix of shape {self.core_capacities.shape}"
  
  
  @property
  def n_cores(self):
    return len(self.core_capacities)