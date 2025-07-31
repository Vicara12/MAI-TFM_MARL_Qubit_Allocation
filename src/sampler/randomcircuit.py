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
  

  def sampleBatch(self, batch_size: int) -> Tuple[Tuple[Tuple[Tuple[int, int], ...], ...], torch.Tensor]:
    """
    Generates a batch of random quantum circuits and their corresponding adjacency matrices.

    Args:
      batch_size (int): The number of samples (circuits) to generate in the batch.

    Returns:
      Tuple[
        Tuple[Tuple[Tuple[int, int], ...], ...],
        torch.Tensor
      ]:
        - A tuple containing the batch of gates for each circuit. Each circuit is represented as a tuple of slices,
          where each slice is a tuple of 2-qubit gate pairs (tuples of two integers).
        - A torch.Tensor of shape (batch_size, max_num_slices, num_lq, num_lq) representing the adjacency matrices
          for each circuit in the batch, where max_num_slices is the maximum number of slices among all circuits,
          and num_lq is the number of logical qubits.

    Notes:
      - If the number of slices per circuit varies, circuits with fewer slices are padded with empty slices.
      - Each slice contains non-overlapping 2-qubit gates, and the adjacency matrix is symmetric.
    """
    #TODO: this is not efficient, we should speed it up with Cython or similar
    #TODO: adjency matrix can take a lot of memory, we should consider using sparse matrices (implies many changes)

    if callable(self.num_slices):
      num_slices_list = [self.num_slices() for _ in range(batch_size)]
    else:
      num_slices_list = [self.num_slices] * batch_size

    max_num_slices = max(num_slices_list)
    batch_gates = []
    # batch_size x max_num_slices x num_lq x num_lq
    batch_matrices = torch.zeros((batch_size, max_num_slices, self.num_lq_, self.num_lq_), dtype=torch.float32)

    for batch_idx in range(batch_size):
      int_num_slices = num_slices_list[batch_idx]
      circuit_slice_gates = []
      for t in range(int_num_slices):
        used_qubits = set()
        slice_gates = []
        while True:
          a, b = random.sample(range(0, self.num_lq_), 2)
          if a in used_qubits or b in used_qubits:
            break
          batch_matrices[batch_idx, t, a, b] = batch_matrices[batch_idx, t, b, a] = 1
          slice_gates.append((a, b))
          used_qubits.add(a)
          used_qubits.add(b)
        circuit_slice_gates.append(tuple(slice_gates))

      while len(circuit_slice_gates) < max_num_slices:
        circuit_slice_gates.append(tuple())
      batch_gates.append(tuple(circuit_slice_gates))

    return tuple(batch_gates), batch_matrices
  
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
    
  
  def __str__(self):
    if self.num_slices is Callable:
      ns = str(self.num_slices())
    else:
      ns = str(self.num_slices)
    return f"RandomCircuit(num_lq={self.num_lq}, num_slices={ns})"