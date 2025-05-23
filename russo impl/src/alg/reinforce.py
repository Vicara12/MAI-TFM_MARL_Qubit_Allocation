import copy
import torch.optim as optim
from scipy import stats
from alg.qubitallocator import QubitAllocator
from utils.allocationcost import cost
from sampler.circuitsampler import CircuitSampler


class Reinforce:
  ''' REINFORCE with Rollout Baseline to train circuit slicer.

  This class implements the algorithm described in Ref. [1] and uses it to train the circuit slicing
  model described in Ref. [2].

  Args:
    - circuit_sampler: Object of a class derived from CircuitSampler
    - qubit_allocator: QubitAllocator object.

  References:
    [Attention, Learn to Solve Routing Problems!]
    (https://arxiv.org/abs/1803.08475).
      Wouter Kool, Herke van Hoof, Max Welling. 2019.

    [Attention-Based Deep Reinforcement Learning for Qubit Allocation in Modular Quantum Architectures]
    (https://arxiv.org/abs/2406.11452).
      Enrico Russo, Maurizio Palesi, Davide Patti, Giuseppe Ascia, Vincenzo Catania. 2024.    
  '''
  def __init__(self, circuit_sampler: CircuitSampler,
               qubit_allocator: QubitAllocator):
    self.circuit_sampler = circuit_sampler
    self.qubit_allocator = qubit_allocator
  

  def _predictBoth(self, qubit_allocator_BL: QubitAllocator, use_greedy: bool):
    circuit_slice_gates, circuit_slice_matrices = self.circuit_sampler.sample()
    allocs, log_probs = self.qubit_allocator(circuit_slice_gates, circuit_slice_matrices, greedy = use_greedy)
    allocs_BL, _ = qubit_allocator_BL(circuit_slice_gates, circuit_slice_matrices, greedy = True)
    R = cost(allocs, self.qubit_allocator.core_con)
    R_BL = cost(allocs_BL, qubit_allocator_BL.core_con)
    return R, R_BL, log_probs


  def train(self, epochs: int,
                  steps: int,
                  batch_size: int,
                  repl_significance: float,
                  lr: float):
    ''' Train the qubit_allocator object.

    This function assumes that the model is already at the target device.

    Args:
      - epochs: number of epochs (E) used for training.
      - steps: number of steps (T) per epoch.
      - batch_size: batch size (B) used when training.
      - repl_significance: significance of the t-test (alpha) required for replacing the baseline policy.
      - lr: learning rate.
    '''
    qubit_allocator_BL = copy.deepcopy(self.qubit_allocator)
    qubit_allocator_BL.eval() # We want this model to be fixed
    opt = optim.Adam(self.qubit_allocator.parameters(), lr=lr)

    for e in range(epochs):
      self.qubit_allocator.train()
      for t in range(steps):
        loss = 0
        for _ in range(batch_size):
          R, R_BL,log_probs = self._predictBoth(qubit_allocator_BL, use_greedy=False)
          for log_prob in log_probs:
            loss += (R - R_BL)*log_prob
        opt.zero_grad()
        loss.backward()
        opt.step()
      all_R = []
      all_R_bl = []
      self.qubit_allocator.eval()
      for _ in range(100):
        R, R_BL,_ = self._predictBoth(qubit_allocator_BL, use_greedy=True)
        all_R.append(R)
        all_R_bl.append(R_BL)
      _, p_value = stats.ttest_rel(all_R, all_R_bl, alternative='greater')
      if p_value < repl_significance:
        self.qubit_allocator = copy.deepcopy(qubit_allocator_BL)
    pass