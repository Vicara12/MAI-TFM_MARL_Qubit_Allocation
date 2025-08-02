from dataclasses import dataclass
from sampler.circuitsampler import CircuitSampler
from qalloczero.alg.mcts import MCTS


class MuZero:
  @dataclass
  class MuZeroTrainParams:
    num_actors: int
    train_iters: int
    sampler: CircuitSampler

  @dataclass
  class MuzeroTrainState:
    pass

  def __init__(self):
    pass

  def optimizeCircuit(self, train_cfg: MuZeroTrainParams) -> None:
    circuit = train_cfg.sampler.sample()
    circuit_embeddings = self.encodeCircuit(circuit)
    allocation = 

  def train(self, train_cfg: MuZeroTrainParams) -> None:
    for train_i in range(train_cfg.train_iters):
      print(f" [*] train_{train_i}")
      mcts_cfg = MCTS.Config()
      for act_i in range(train_cfg.num_actors):
        self.optimizeCircuit(train_cfg)
      updateModels()