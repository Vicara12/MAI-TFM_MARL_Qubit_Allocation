import torch
from typing import Tuple, List
from dataclasses import dataclass
from sampler.circuitsampler import CircuitSampler
from utils.customtypes import Circuit, Hardware
from qalloczero.models.enccircuit import GNNEncoder
from qalloczero.alg.mcts import MCTS
from utils.environment import QubitAllocationEnvironment



class AlphaZero:
  @dataclass
  class TrainConfig:
    train_iters: int
    sampler: CircuitSampler


  @dataclass
  class Config:
    hardware: Hardware
    # First item of encoder_shape determines qubit embedding size and last item circuit emb. size
    encoder_shape: Tuple[int]
    mcts_tree_size: int


  def __init__(self, config: Config, qubit_embs: torch.Tensor):
    self.cfg = config
    self.circuit_encoder = GNNEncoder(hardware=self.cfg.hardware,
                                      nn_dims=self.cfg.encoder_shape,
                                      qubit_embs=qubit_embs)


  def optimizeCircuit(self, circuit: Circuit) -> Tuple[torch.Tensor, List]:
    circuit_embeddings = self.circuit_encoder.encodeCircuits([circuit])[0]
    env = QubitAllocationEnvironment(circuit=circuit, hardware=self.cfg.hardware)
    mcts = MCTS(circuit_embs=circuit_embeddings,
                circuit=circuit,
                hardware=self.cfg.hardware,
                config=MCTS.Config() # Default config is good for now
              )
    action_history = []

    # Run MCTS
    for alloc_step in circuit.alloc_steps:
      (_, qubits_step) = alloc_step
      alloc_to_core, n_sims = mcts.iterate(self.cfg.mcts_tree_size)
      total_cost = 0
      for qubit in qubits_step:
        total_cost += env.allocate(alloc_to_core, qubit)
      action_history.append([alloc_step, alloc_to_core, total_cost])
      print(f" [{list(alloc_step)} -> {alloc_to_core}] sims={n_sims} c={total_cost}")
    
    assert env.finished, "did not finished optimizing circuit!"

    # Compute V for each action i by adding the total allocation cost from i until the end
    # Iterate the list of actions backwards ignoring the last item
    for i in range(len(action_history)-2,-1,-1):
      action_history[i][2] += action_history[i+1][2]
        
    return env.qubit_allocations, action_history


  def train(self, train_cfg: TrainConfig) -> None:
    for train_i in range(train_cfg.train_iters):
      print(f" [*] train_{train_i}")
      circuit = train_cfg.sampler.sample()
      env, action_history = self.optimizeCircuit(circuit=circuit)
      # updateModels()