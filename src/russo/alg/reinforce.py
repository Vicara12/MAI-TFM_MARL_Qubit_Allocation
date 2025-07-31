import copy
import os
from math import log10, floor
import torch.optim as optim
from scipy import stats
from typing import Union
import matplotlib.pyplot as plt
from russo.alg.qubitallocator import QubitAllocator
from utils.allocutils import cost
from utils.modelutils import genTrainFolder, getTrainFolderPath
from utils.plotter import drawCircuit, drawQubitAllocation
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
               qubit_allocator: QubitAllocator,
               train_folder: Union[str, None]=None):
    self.circuit_sampler = circuit_sampler
    self.qubit_allocator = qubit_allocator
    self.train_path = getTrainFolderPath("russo")
    self.train_folder = train_folder
  
  def sampleCircuitBatch(self, batch_size, device: str):
    ''' Samples a batch of circuits and return the circuit slice gates and matrices. '''
    circuit_slice_gates, circuit_slice_matrices = self.circuit_sampler.sampleBatch(batch_size)
    circuit_slice_matrices = circuit_slice_matrices.to(device)
    return circuit_slice_gates, circuit_slice_matrices
  
  def _predictBoth(self, qubit_allocator_BL: QubitAllocator, batch_size=256, use_greedy: bool=False, device: str='cuda'):
    #circuit_slice_gates, circuit_slice_matrices = self.circuit_sampler.sample()
    #circuit_slice_matrices = circuit_slice_matrices.to(device)
    circuit_slice_gates, circuit_slice_matrices = self.sampleCircuitBatch(batch_size=batch_size, device=device)
    allocs, log_probs = self.qubit_allocator(circuit_slice_gates, circuit_slice_matrices, greedy = use_greedy)
    allocs_BL, _ = qubit_allocator_BL(circuit_slice_gates, circuit_slice_matrices, greedy = True)
    R = cost(allocs, self.qubit_allocator.core_con)
    R_BL = cost(allocs_BL, qubit_allocator_BL.core_con)
    return R, R_BL, log_probs
  

  def _trainBatch(self, batch_size: int, qubit_allocator_BL: QubitAllocator, device: str, opt: optim.Adam):
    R, R_BL,log_probs = self._predictBoth(qubit_allocator_BL, batch_size=batch_size, use_greedy=False, device=device)
    advantage = R - R_BL # [batch]
    loss = (advantage * log_probs.sum(dim=1)).mean() # sum log probs for all actions in the trajectory (T*Q)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return R, R_BL
  

  def _validation(self, num_val_runs: int, qubit_allocator_BL: QubitAllocator, device: str):
    self.qubit_allocator.eval()
    R, R_BL,_ = self._predictBoth(qubit_allocator_BL, batch_size=num_val_runs, use_greedy=False, device=device)
    self.qubit_allocator.train()
    return R, R_BL


  def _genCheckpoint(self, subfolder: Union[str, None], device: str):
    if self.train_folder is None:
      self.train_folder = genTrainFolder(self.qubit_allocator.num_lq, "russo")
    if subfolder is None:
      folder = os.path.join(self.train_path, self.train_folder)
    else:
      folder = os.path.join(self.train_path, self.train_folder, subfolder)
    os.makedirs(folder)
    # Save model
    torch.save(self.qubit_allocator, os.path.join(folder, "model.pth"))
    # Generate a circuit and allocate qubits as a test
    self.qubit_allocator.eval()
    circuit_slice_gates, circuit_slice_matrices = self.circuit_sampler.sampleBatch(batch_size=1)
    circuit_slice_matrices = circuit_slice_matrices.to(device)
    allocs, _ = self.qubit_allocator(circuit_slice_gates, circuit_slice_matrices, greedy=True)
    R = cost(allocs, self.qubit_allocator.core_con)
    # Save figures of qubit allocations
    plt.clf()
    drawCircuit(circuit_slice_gates, self.qubit_allocator.num_lq, show=False)
    plt.savefig(os.path.join(folder, "circuit.svg"))
    plt.clf()
    drawQubitAllocation(allocs.cpu(), self.qubit_allocator.core_capacities.cpu(), circuit_slice_gates, show=False)
    plt.savefig(os.path.join(folder, f"allocations_cost_{R}.svg"))
    plt.clf()


  def train(self, epochs: int,
                  steps: int,
                  batch_size: int,
                  repl_significance: float,
                  lr: float,
                  num_val_runs: int = 20,
                  verbose: bool = True,
                  checkpoint_each: Union[None, int] = None,
                  save_at_end: bool = True):
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
    #torch.autograd.set_detect_anomaly(True)

    device = next(self.qubit_allocator.parameters()).device
    qubit_allocator_BL = copy.deepcopy(self.qubit_allocator)
    qubit_allocator_BL.eval() # We want this model to be fixed
    opt = optim.Adam(self.qubit_allocator.parameters(), lr=lr)

    l_ep = 1+floor(log10(epochs)) # number of digits in epochs
    l_st = 1+floor(log10(steps))  # number of digits in steps
    res_format = lambda Rs, Rbls: f"R={Rs.mean():.02f} R_bl={Rbls.mean():.02f}"
    fmt_step = lambda t: f"{t+1:{l_st}d}/{steps}"
    fmt_res = lambda Rs, Rbls, e, t_str: f"[{e+1:{l_ep}d}/{epochs:},{t_str}] {res_format(Rs, Rbls)}"

    history_train = []
    history_val = []

    try:
      for e in range(epochs):
        self.qubit_allocator.train()
        epoch_history_train = []
        for t in range(steps):
          R, R_bl = self._trainBatch(batch_size=batch_size,
                                            qubit_allocator_BL=qubit_allocator_BL,
                                            device=device,
                                            opt=opt)
          if verbose:
            print(fmt_res(R, R_bl, e, fmt_step(t)))
          epoch_history_train.append(R)
        R, R_bl = self._validation(num_val_runs=num_val_runs,
                                          qubit_allocator_BL=qubit_allocator_BL,
                                          device=device)
        history_train.append(epoch_history_train)
        history_val.append(sum(R)/len(R))
        _, p_value = stats.ttest_rel(R.cpu(), R_bl.cpu(), alternative='less')
        if p_value < repl_significance:
          qubit_allocator_BL = copy.deepcopy(self.qubit_allocator)
        if verbose:
          print(f"{fmt_res(R, R_bl, e, 'val')}, p_val={p_value:.3f} ({repl_significance} {'updating BL' if p_value < repl_significance else 'keep BL'})")
        if checkpoint_each is not None and (e+1)%checkpoint_each == 0:
          self._genCheckpoint(subfolder=f"{e}_R_{int(sum(R)/len(R))}", device=device)
    except KeyboardInterrupt:
      pass

    if save_at_end:
      self._genCheckpoint(subfolder=None, device=device)
    
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