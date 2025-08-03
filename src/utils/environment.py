from typing import Tuple
import torch
from utils.customtypes import Circuit, Hardware


class QubitAllocationEnvironment:
  def __init__(self, circuit: Circuit, hardware: Hardware):
    self.circuit = circuit
    self.hardware = hardware
    self.allocations = torch.empty(shape=(self.circuit.n_slices, self.circuit.n_qubits), dtype=int)
    self.reset()
  

  def reset(self):
    self.current_core_caps = self.hardware.core_capacities
    self.current_slice = 0
    self.qubit_is_allocated = torch.tensor([False]*self.circuit.n_qubits, dtype=bool)
    self.qubits_to_allocate = self.circuit.n_qubits
  
  
  def allocate(self, core: int, qubit: int) -> int:
    ''' Assign a core to a qubit in the current time slice and returns the cost of the allocation.

    This is the only action in the environment. Contains several asserts to ensure the validity of
    the solution.
    '''
    assert self.current_slice < self.circuit.n_slices, "Tried to allocate past the end of the circuit"
    assert core >= 0 and core < self.hardware.n_cores, \
      f"Tried to allocate to core {core} not in [0,{self.hardware.n_cores-1}]"
    assert qubit >= 0 and qubit < self.circuit.n_qubits, \
      f"Tried to allocate qubit {qubit} not in [0,{self.circuit.n_qubits-1}]"
    assert not self.qubit_is_allocated[qubit], f"Tried to allocate qubit {qubit} twice"
    assert self.hardware.core_capacities[core] > 0, f"Tried to allocate to complete core {core}"

    self.allocations[qubit, self.current_slice] = core
    self.qubit_is_allocated[qubit] = True
    self.qubits_to_allocate -= 1

    # If finished allocation of time slice
    if self.qubits_to_allocate == 0:
      # Check all gates have their qubits in the same core
      for gate in self.circuit.slice_gates[self.current_slice]:
        assert self.allocations[gate[0], self.current_slice] == self.allocations[gate[1], self.current_slice], \
          (f"In time slice {self.current_slice} allocated qubit {gate[0]} and {gate[0]} to cores "
           f"{self.allocations[gate[0], self.current_slice]} and "
           f"{self.allocations[gate[1], self.current_slice]}, but they belong to the same gate")
        
        self.qubit_is_allocated = torch.tensor([False]*self.circuit.n_qubits, dtype=bool)
        self.qubits_to_allocate = self.circuit.n_qubits
        self.current_slice += 1

    # Return cost of the allocation
    if self.current_slice == 0:
      return 0
    prev_core = self.allocations[qubit, self.current_slice-1]
    return self.hardware.core_connectivity[prev_core,core]
        

  @property
  def current_slice(self) -> int:
    return self.current_slice
  

  @property
  def finished(self) -> bool:
    return self.current_slice == self.circuit.n_slices
  

  @property
  def prev_slice_allocations(self) -> torch.Tensor:
    assert self.current_slice > 0, "No previous slice"
    return self.allocations[:,self.current_slice-1].squeeze()
  

  @property
  def qubit_allocations(self) -> torch.Tensor:
    assert self.finished, "Tried to get incomplete allocation list"
    return self.allocations