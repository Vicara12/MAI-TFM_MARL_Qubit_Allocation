import torch
from utils.timer import Timer
from sampler.randomcircuit import RandomCircuit
from qalloczero.models.enccircuit import GNNEncoder
from utils.circuitutils import getCircuitMatrices2xE
from utils.customtypes import Hardware


def main():
  sampler = RandomCircuit(num_lq=5, num_slices=2)
  circuits = [sampler.sample() for _ in range(2)]
  matrices = [getCircuitMatrices2xE(c) for c in circuits]
  hardware = Hardware(core_capacities=(5,5), core_connectivity=torch.Tensor([[0,1],[1,0]]))
  circuit_encoder = GNNEncoder(hardware=hardware, nn_dims=[8,4,4])
  t1 = Timer('t1')
  t2 = Timer('t2')
  # TODO: test
  with t1:
    embs = circuit_encoder.encodeCircuits(circuits)
  # embs_ = []
  # for matrix in matrices:
  #   with t2:
  #     e = circuit_encoder([matrix])
  #   embs_.append(e)
  # embs_ = torch.concat(embs_, axis= 0)
  # print(f"t_bat={t1.total_time}\nt_sep={t2.total_time}")
  print(embs)


if __name__ == "__main__":
  main()