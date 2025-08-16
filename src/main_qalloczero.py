import torch
from utils.timer import Timer
from utils.allocutils import solutionCost
from sampler.randomcircuit import RandomCircuit
from qalloczero.alg.alphazero import AlphaZero
from qalloczero.models.snapshotenc import SnapEncModel
from qalloczero.models.predmodel import PredictionModel
from qalloczero.models.inferenceserver import InferenceServer
from utils.customtypes import Hardware
from utils.plotter import drawQubitAllocation



def main():
  core_caps = torch.tensor([4,4,4], dtype=int)
  core_con = torch.ones(size=(len(core_caps),len(core_caps)), dtype=int) - torch.eye(n=len(core_caps), dtype=int)
  hardware = Hardware(core_capacities=core_caps, core_connectivity=core_con)
  q_emb_size = 16
  q_embs = torch.nn.Parameter(torch.randn(hardware.n_physical_qubits, q_emb_size), requires_grad=True)
  dummy_q_emb = torch.nn.Parameter(torch.randn(q_emb_size), requires_grad=True)
  snap_enc = SnapEncModel(
    nn_dims=(16,8),
    hardware=hardware,
    qubit_embs=q_embs,
    dummy_qubit_emb=dummy_q_emb
  )
  pred_mod = PredictionModel(
     config=PredictionModel.Config(hw=hardware, circuit_emb_shape=8, mha_num_heads=4),
     qubit_embs=q_embs
  )

  InferenceServer.addModel("snap_enc_model", snap_enc, unpack=True)
  InferenceServer.addModel("pred_model", pred_mod, unpack=False)

  circuit = RandomCircuit(num_lq=sum(core_caps).item(), num_slices=30).sample()

  azero_config = AlphaZero.Config(hardware=hardware, encoder_shape=(16,8,8), mcts_tree_size=1024)
  azero = AlphaZero(config=azero_config, qubit_embs=q_embs)
  with Timer.get('t0'):
    allocations, history = azero.optimizeCircuit(circuit=circuit)
  print("\nHistory:")
  for i, piece in enumerate(history):
     print(f"{i}: {piece}")

  cost = solutionCost(allocations,hardware.core_connectivity)
  print(f" -> cost={cost} time={Timer.get('t0').total_time}")
  drawQubitAllocation(allocations, core_caps, circuit.slice_gates)


if __name__ == "__main__":
    main()