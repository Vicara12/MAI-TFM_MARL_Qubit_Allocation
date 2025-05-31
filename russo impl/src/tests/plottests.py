import torch
from sampler.randomcircuit import RandomCircuit
from utils.plotter import drawCircuit, drawQubitAllocation
from utils.allocutils import validate

def plottingDemo():
  num_lq = 8
  num_slices = 6
  sampler = RandomCircuit(num_lq=num_lq, num_slices=num_slices)
  circuit_tuples, _ = sampler.sample()
  drawCircuit(circuit_tuples, num_lq=num_lq)

  allocation = (
    (0, 1, 2, 3),
    (2, 1, 3, 0),
    (2, 3, 0, 1),
  )
  allocation = torch.Tensor(allocation).T
  circuit_slice_gates = (((0,1),),((0,3),),((0,1),))
  core_capacities = (1,1,2)
  print(f"Solution is {'valid' if validate(allocation, circuit_slice_gates, core_capacities) else 'not valid'}")
  drawQubitAllocation(allocation,
                      core_capacities=core_capacities,
                      circuit_slice_gates=circuit_slice_gates)