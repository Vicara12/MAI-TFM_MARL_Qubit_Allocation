from fgproee.alg.fgp import FineGrainedPartitionerROEE
from sampler.randomcircuit import RandomCircuit
from tomas.env import QubitAllocationEnv
from fgproee.alg.fgp import buildLookaheadWeights



def basicTrainParams():
  return dict(
    num_lq = 8,
    num_slices = 2,
    num_cores = 2,
    use_extended = False,
    batch_size = 1,
    N_max = 50,
    alpha = 0.1,
    beta = 0.1,
    gamma = 0.9,
    initial_capacity = 4,
    seed = 0
  )



def trainDemo(num_lq: int,
              num_slices: int,
              num_cores: int,
              initial_capacity: int,
              batch_size: int,
              use_extended: bool = False,
              N_max: int = 50,
              alpha: float = 3,
              beta: float = 0.1,
              gamma: float = 0.9,
              seed: int = 0,
            ):
    
    sampler = RandomCircuit(num_lq=num_lq, num_slices=num_slices)
    _, A1 = sampler.sampleBatch(batch_size=batch_size)
    A1 = A1.squeeze(0)  

    # Get lookahead weights
    A2 = buildLookaheadWeights(A1, sigma=1.0)
    env = QubitAllocationEnv(N_max=N_max,
                 C=num_cores,
                 use_extended=False,
                 alpha=alpha,
                 beta=beta,
                 gamma=gamma,
                 initial_capacity=initial_capacity)
    obs, info = env.reset(options={'N': num_lq, 
                                   'Madj': A1.cpu().numpy(), 
                                   'Mweights': A2.cpu().numpy(), 
                                   'T': num_slices})
    env.render()
    action = env.action_space.sample()  # Random action
    next_obs, reward, terminated, truncated, info = env.step(action)
