from typing import List
import torch
from utils.customtypes import Circuit



def getCircuitMatricesNxN(circuit: Circuit) -> torch.Tensor:
  ''' Return the circuit slice matrices from circuit_slice_gates.

  circuit_slice_matrices is a rank 3 tensor, effectively a vector with a matrix for each time
  slice in the circuit. This matrix contains a 1 at position (i,j) if there is a gate between
  qubits i and j in that time slice, zero otherwise.
  '''
  matrices = torch.zeros(size=[circuit.n_slices, circuit.n_qubits, circuit.n_qubits], dtype=int)
  for slice_i, slice in enumerate(circuit.slice_gates):
    for gate in slice:
      matrices[slice_i, gate[0], gate[1]] = matrices[slice_i, gate[1], gate[0]] = 1
  return matrices


def getCircuitMatrices2xE(circuit: Circuit) -> List[torch.Tensor]:
  ''' Convert slices into custom type used by pytorch_geometric GNN class GCNConv.

  The custom type encodes graphs as a 2xE matrix, in which items at positions 0xi and 1xi indicate
  the origin and destination of the ith edge in a directed graph. This method returns a list of
  graphs encoded as described above in which each graph corresponds to one time slice.
  '''
  tensors = []
  for slice in circuit.slice_gates:
    slice_tensor = torch.empty(size=(2,2*len(slice)), dtype=int)
    for gate_i, gate in enumerate(slice):
      # Insert edge twice as a->b and a<-b to get a<->b
      slice_tensor[0, 2*gate_i] = slice_tensor[1, 2*gate_i+1] = gate[0]
      slice_tensor[1, 2*gate_i] = slice_tensor[0, 2*gate_i+1] = gate[1]
    tensors.append(slice_tensor)
  return tensors