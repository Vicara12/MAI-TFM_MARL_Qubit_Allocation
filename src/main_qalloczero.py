import torch
from utils.timer import Timer
from utils.allocutils import solutionCost, validate
from sampler.randomcircuit import RandomCircuit
from qalloczero.alg.alphazero import AlphaZero
from qalloczero.models.predmodel import PredictionModel
from qalloczero.models.inferenceserver import InferenceServer
from utils.customtypes import Hardware
from utils.plotter import drawQubitAllocation



def main():
  core_caps = torch.tensor([4,4,4,4], dtype=int)
  core_con = torch.ones(size=(len(core_caps),len(core_caps)), dtype=int) - torch.eye(n=len(core_caps), dtype=int)
  hardware = Hardware(core_capacities=core_caps, core_connectivity=core_con)
  q_embs = torch.nn.Parameter(torch.randn(hardware.n_physical_qubits, 2), requires_grad=True)
  pred_mod = PredictionModel(
     config=PredictionModel.Config(hw=hardware, mha_num_heads=4),
     qubit_embs=q_embs
  )
  InferenceServer.setModel(pred_mod)

  circuit = RandomCircuit(num_lq=sum(core_caps).item(), num_slices=30).sample()

  azero_config = AlphaZero.Config(hardware=hardware, encoder_shape=(2,8,8), mcts_tree_size=256)
  azero = AlphaZero(config=azero_config, qubit_embs=q_embs)
  allocations, history = azero.optimizeCircuit(circuit=circuit)
  print("\nHistory:")
  for piece in history:
     print(piece)

  cost = solutionCost(allocations,hardware.core_connectivity)
  valid = validate(allocations,hardware.core_connectivity, hardware.core_capacities)
  print(f" -> valid={valid}, cost={cost}")
  drawQubitAllocation(allocations, core_caps, circuit.slice_gates)


if __name__ == "__main__":
    main()