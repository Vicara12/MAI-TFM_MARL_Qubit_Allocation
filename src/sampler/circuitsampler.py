from abc import abstractmethod, ABC
from typing import Tuple
import torch


class CircuitSampler(ABC):
  def __init__(self, num_lq):
    self.num_lq_ = num_lq
  
  @property
  def num_lq(self):
    return self.num_lq_
  
  @abstractmethod
  def sample(self) -> Tuple[ Tuple[Tuple[Tuple[int,int], ...], ...], torch.Tensor ]:
    ''' Function that takes no arguments and returns a problem sample.

    A problem instance is composed of a circuit_slice_gates tuple and a circuit_slice_matrices tuple.
    circuit_slice_gates is a tuple that contains a tuple for each time slice in the circuit. This
    time slice tuple is then made of another tuple per gate in that time slice, which contains the
    index (starting from 0) of the two qubits that form the gate.
    circuit_slice_matrices is a rank 3 tensor, effectively a vector with a matrix for each time
    slice in the circuit. This matrix contains a 1 at position (i,j) if there is a gate between
    qubits i and j in that time slice, zero otherwise.

    Returns:
      - circuit_slice_gates
      - circuit_slice_matrices
    '''
    raise NotImplementedError("implement sampler for class")
  
  @abstractmethod
  def __str__(self) -> str:
    raise NotImplementedError("implement __str__ method for class")