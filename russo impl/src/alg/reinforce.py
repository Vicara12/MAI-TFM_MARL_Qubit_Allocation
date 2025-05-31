import copy
from math import log10, floor
import torch.optim as optim
from scipy import stats
from alg.qubitallocator import QubitAllocator
from utils.allocutils import cost
from sampler.circuitsampler import CircuitSampler


import torch # TODO REMOVE

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
  

  def _predictBoth(self, qubit_allocator_BL: QubitAllocator, use_greedy: bool, device: str):
    circuit_slice_gates, circuit_slice_matrices = self.circuit_sampler.sample()
    circuit_slice_matrices = circuit_slice_matrices.to(device)
    allocs, log_probs = self.qubit_allocator(circuit_slice_gates, circuit_slice_matrices, greedy = use_greedy)
    allocs_BL, _ = qubit_allocator_BL(circuit_slice_gates, circuit_slice_matrices, greedy = True)
    R = cost(allocs, self.qubit_allocator.core_con)
    R_BL = cost(allocs_BL, qubit_allocator_BL.core_con)
    return R, R_BL, log_probs
  

  def _trainBatch(self, batch_size: int, qubit_allocator_BL: QubitAllocator, device: str, opt: optim.Adam):
    loss = 0
    all_R = []
    all_R_bl = []
    for b in range(batch_size):
      R, R_BL,log_probs = self._predictBoth(qubit_allocator_BL, use_greedy=False, device=device)
      all_R.append(R)
      all_R_bl.append(R_BL)
      for log_prob in log_probs:
        loss += (R - R_BL)*log_prob
    opt.zero_grad()
    loss.backward()
    opt.step()
    return all_R, all_R_bl
  

  def _validation(self, num_val_runs: int, qubit_allocator_BL: QubitAllocator, device: str):
    all_R = []
    all_R_bl = []
    self.qubit_allocator.eval()
    for v in range(num_val_runs):
      R, R_BL,_ = self._predictBoth(qubit_allocator_BL, use_greedy=True, device=device)
      all_R.append(R)
      all_R_bl.append(R_BL)
    return all_R, all_R_bl
  

  def train(self, epochs: int,
                  steps: int,
                  batch_size: int,
                  repl_significance: float,
                  lr: float,
                  num_val_runs: int = 20,
                  verbose: bool = True):
    ''' Train the qubit_allocator object.

    This function assumes that the model is already at the target device.

    Args:
      - epochs: number of epochs (E) used for training.
      - steps: number of steps (T) per epoch.
      - batch_size: batch size (B) used when training.
      - repl_significance: significance of the t-test (alpha) required for replacing the baseline policy.
      - lr: learning rate.
      - num_val_runs: number of validation runs when checking for model update.
    '''
    device = next(self.qubit_allocator.parameters()).device
    qubit_allocator_BL = copy.deepcopy(self.qubit_allocator)
    qubit_allocator_BL.eval() # We want this model to be fixed
    opt = optim.Adam(self.qubit_allocator.parameters(), lr=lr)

    l_ep = 1+floor(log10(epochs)) # number of digits in epochs
    l_st = 1+floor(log10(steps))  # number of digits in steps
    res_format = lambda Rs, Rbls: f"R={sum(Rs)/len(Rs):.02f} R_bl={sum(Rbls)/len(Rbls):.02f}"
    fmt_step = lambda t: f"{t+1:{l_st}d}/{steps}"
    fmt_res = lambda Rs, Rbls, e, t_str: f"[{e+1:{l_ep}d}/{epochs:},{t_str}] {res_format(Rs, Rbls)}"

    history_train = []
    history_val = []

    try:
      for e in range(epochs):
        self.qubit_allocator.train()
        epoch_history_train = []
        for t in range(steps):
          all_R, all_R_bl = self._trainBatch(batch_size=batch_size,
                                            qubit_allocator_BL=qubit_allocator_BL,
                                            device=device,
                                            opt=opt)
          if verbose:
            print(fmt_res(all_R, all_R_bl, e, fmt_step(t)))
          epoch_history_train.append(sum(all_R)/len(all_R))
        all_R, all_R_bl = self._validation(num_val_runs=num_val_runs,
                                          qubit_allocator_BL=qubit_allocator_BL,
                                          device=device)
        history_train.append(epoch_history_train)
        history_val.append(sum(all_R)/len(all_R))
        _, p_value = stats.ttest_rel(all_R, all_R_bl, alternative='less')
        if p_value < repl_significance:
          qubit_allocator_BL = copy.deepcopy(self.qubit_allocator)
        if verbose:
          print(f"{fmt_res(all_R, all_R_bl, e, "val")}, p_val={p_value:.3f} ({repl_significance} {'updating BL' if p_value < repl_significance else 'keep BL'})")
    except KeyboardInterrupt:
      pass
    
    return dict(
      train_params = dict(
        epochs=epochs,
        steps=steps,
        batch_size=batch_size,
        repl_significance=repl_significance,
        lr=lr,
        num_val_runs=num_val_runs
      ),
      qubit_allocator=str(self.qubit_allocator),
      history_train=history_train,
      history_val=history_val
    )