import random
import torch
from typing import Tuple, Union, Callable
from sampler.circuitsampler import CircuitSampler


class RandomCircuit(CircuitSampler):
  ''' Random Circuit Sampler.

  Args:
    - num_lq: number of logical qubits of the circuit.
    - num_slices: number of slices of the generated circuit. If set to int its value is fixed, if
        set to a callable object then the number of slices each time sample() is called is determined
        by the returned value (i.e. lambda: randint(1,10)).
  '''
  def __init__(self, num_lq: int, num_slices: Union[int, Callable[[], int]]):
    super().__init__(num_lq)
    self.num_slices = num_slices
  

  def sample(self) -> Tuple[ Tuple[Tuple[Tuple[int,int], ...], ...], torch.Tensor ]:
    int_num_slices = self.num_slices() if self.num_slices is Callable else self.num_slices
    circuit_slice_matrices = torch.zeros(size=(int_num_slices, self.num_lq_, self.num_lq_))
    circuit_slice_gates = []
    a,b = random.sample(range(0,self.num_lq_),2)
    for t in range(int_num_slices):
      used_qubits = set()
      slice_gates = []
      while not (a in used_qubits or b in used_qubits):
        circuit_slice_matrices[t,a,b] = circuit_slice_matrices[t,b,a] = 1
        slice_gates.append((a,b))
        used_qubits.add(a)
        used_qubits.add(b)
        a,b = random.sample(range(0,self.num_lq_),2)
      circuit_slice_gates.append(tuple(slice_gates))
    return tuple(circuit_slice_gates), circuit_slice_matrices