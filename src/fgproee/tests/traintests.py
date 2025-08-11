from fgproee.alg.fgp import FineGrainedPartitionerROEE
from sampler.randomcircuit import RandomCircuit



def basicTrainParams():
  return dict(
    num_lq = 16,
    num_slices = 15,
    num_cores = 4,
    max_oee_passes = 100,
    sigma = 1.0,
  )



def trainDemo(num_lq: int,
              num_slices: int,
              num_cores: int,
              max_oee_passes: int = 100,
              sigma: float = 1.0):
    
    sampler = RandomCircuit(num_lq=num_lq, num_slices=num_slices)
    _, circuit_slice_matrices = sampler.sampleBatch(batch_size=1)
    circuit_slice_matrices = circuit_slice_matrices.squeeze(0)  # Remove batch dimension

    # According to FGPrOEE, we use sigma=1.0 for lookahead weights
    fg = FineGrainedPartitionerROEE(k=num_cores, sigma=sigma, max_oee_passes=max_oee_passes,seed=42)
    maps, costs = fg.run(circuit_slice_matrices, verbose=True)
    print("Per-slice costs:", costs)
