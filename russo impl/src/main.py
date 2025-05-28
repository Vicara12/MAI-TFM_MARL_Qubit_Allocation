import torch
from alg.qubitallocator import QubitAllocator
from alg.reinforce import Reinforce
from sampler.randomcircuit import RandomCircuit
from utils.plotter import drawCircuit, drawQubitAllocation
from utils.allocutils import validate


def plottingDemo():
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
  allocation = torch.Tensor(allocation).T
  circuit_slice_gates = (((0,1),),((0,3),),((0,1),))
  core_capacities = (1,1,2)
  print(f"Solution is {'valid' if validate(allocation, circuit_slice_gates, core_capacities) else 'not valid'}")
  drawQubitAllocation(allocation,
                      core_capacities=core_capacities,
                      circuit_slice_gates=circuit_slice_gates)


def trainDemo():
  num_lq = 4
  num_slices = 10
  device='cuda'
  if device == 'cuda' and not torch.cuda.is_available():
    raise Exception("cuda selected but not available")
  random_sampler = RandomCircuit(num_lq=num_lq, num_slices=num_slices)
  allocator = QubitAllocator(num_lq=num_lq,
                             emb_size=32,
                             num_enc_transf=3,
                             num_enc_transf_heads=4,
                             core_con=torch.tensor([[0,1],[1,0]]),
                             core_capacities=torch.tensor([2,2]))
  allocator.to(device=device)
  reinforce = Reinforce(circuit_sampler=random_sampler, qubit_allocator=allocator)
  reinforce.train(epochs=5,
                  steps=5,
                  batch_size=8,
                  repl_significance=0.1,
                  lr=0.0001,
                  num_val_runs=5)



if __name__ == '__main__':
  # plottingDemo()
  trainDemo()