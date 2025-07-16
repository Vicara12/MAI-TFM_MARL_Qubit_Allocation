import torch
from alg.qubitallocator import QubitAllocator
from alg.reinforce import Reinforce
from sampler.randomcircuit import RandomCircuit


def basicTrainParams():
  return dict(
    num_lq = 16,
    num_slices = 15,
    num_cores = 4,
    epochs = 10,
    steps = 1,
    batch_size = 16,
    checkpoint_each = 1,
  )


def trainDemo(num_lq: int,
              num_slices: int,
              num_cores: int,
              epochs: int,
              steps: int,
              batch_size: int,
              checkpoint_each: int):
  core_con = torch.ones((num_cores,num_cores), dtype=int) - torch.eye(num_cores, dtype=int)
  core_capacities = torch.tensor([num_lq//num_cores]*num_cores)
  device='cuda'
  if device == 'cuda' and not torch.cuda.is_available():
    raise Exception("cuda selected but not available")
  random_sampler = RandomCircuit(num_lq=num_lq, num_slices=num_slices)
  allocator = QubitAllocator(num_lq=num_lq,
                             emb_size=256,
                             num_enc_transf=3,
                             num_enc_transf_heads=8,
                             core_con=core_con,
                             core_capacities=core_capacities)
  allocator.to(device=device)
  reinforce = Reinforce(circuit_sampler=random_sampler, qubit_allocator=allocator)
  reinforce.train(epochs=epochs,
                  steps=steps,
                  batch_size=batch_size,
                  repl_significance=0.1,
                  lr=1e-4,
                  num_val_runs=50,
                  checkpoint_each=checkpoint_each,
                  save_at_end=True)
