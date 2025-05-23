from sampler.randomcircuit import RandomCircuit
from utils.plotter import drawCircuit, drawQubitAllocation
import torch


if __name__ == '__main__':
  num_lq = 8
  num_slices = 6
  sampler = RandomCircuit(num_lq=num_lq, num_slices=num_slices)
  circuit_tuples, _ = sampler.sample()
  # drawCircuit(circuit_tuples, num_lq=num_lq)

  allocation = (
    (0, 1, 2, 3),
    (2, 1, 3, 0),
    (2, 3, 0, 1),
  )
  circuit_slice_gates = (((0,1),(2,3)),((0,2),),((2,3),))
  drawQubitAllocation(torch.Tensor(allocation).T,
                      core_sizes=(1,1,2),
                      circuit_slice_gates=circuit_slice_gates)
