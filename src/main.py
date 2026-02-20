import torch
import argparse
from random import randint
from sampler.hardwaresampler import HardwareSampler
from sampler.randomcircuit import RandomCircuit, HotRandomCircuit, DenseRandomCircuit
from sampler.mixedcircuitsampler import MixedCircuitSampler
from qalloczero.alg.directalloc import DirectAllocator
from qalloczero.scripts.test_compare import validate, benchmark, compare_w_sota
from qalloczero.alg.ts import ModelConfigs
from utils.customtypes import Hardware



def train_model_da(allocator, name: str):
  validation_hardware = Hardware(
    core_capacities=torch.tensor([4]*4),
    core_connectivity=(torch.ones(4,4) - torch.eye(4))
  )
  val_sampler = RandomCircuit(num_lq=16, num_slices=32)
  train_cfg = DirectAllocator.TrainConfig(
    train_iters=30_000,
    batch_size=1,
    group_size=32,
    validate_each=25,
    validation_hardware=validation_hardware,
    validation_circuits=[val_sampler.sample() for _ in range(32)],
    store_path=name,
    initial_noise=0.2,
    noise_decrease_factor=0.9995,
    min_noise=0.0,
    circ_sampler=RandomCircuit(num_lq=16, num_slices=(4,32)),
    # circ_sampler=MixedCircuitSampler(num_lq=20, samplers=[
    #   (0.50,      RandomCircuit(num_lq=64, num_slices=lambda: randint(8,64))),
    #   (0.25,   HotRandomCircuit(num_lq=64, num_slices=lambda: randint(8,64))),
    #   (0.25, DenseRandomCircuit(num_lq=64, num_slices=lambda: randint(8,64))),
    # ]),
    lr=5e-5,
    inv_mov_penalization=0.6,
    mask_invalid=False,
    hardware_sampler=HardwareSampler(max_nqubits=16, range_ncores=[2,8]),
    dropout=0.0,
  )
  allocator.train(train_cfg)


def finetune_model_da(name: str, chkpt: int):
  allocator = DirectAllocator.load(name, checkpoint=chkpt).set_mode(DirectAllocator.Mode.Sequential)
  validation_hardware = Hardware(
    core_capacities=torch.tensor([4]*4),
    core_connectivity=(torch.ones(4,4) - torch.eye(4))
  )
  val_sampler = RandomCircuit(num_lq=16, num_slices=32)
  train_cfg = DirectAllocator.TrainConfig(
    train_iters=30_000,
    batch_size=1,
    group_size=32,
    validate_each=25,
    validation_hardware=validation_hardware,
    validation_circuits=[val_sampler.sample() for _ in range(32)],
    store_path=name,
    initial_noise=0.2,
    noise_decrease_factor=0.9995,
    min_noise=0.0,
    circ_sampler=RandomCircuit(num_lq=16, num_slices=(4,32)),
    # circ_sampler=MixedCircuitSampler(num_lq=20, samplers=[
    #   (0.50,      RandomCircuit(num_lq=64, num_slices=lambda: randint(8,64))),
    #   (0.25,   HotRandomCircuit(num_lq=64, num_slices=lambda: randint(8,64))),
    #   (0.25, DenseRandomCircuit(num_lq=64, num_slices=lambda: randint(8,64))),
    # ]),
    lr=5e-5,
    inv_mov_penalization=0.6,
    mask_invalid=False,
    hardware_sampler=HardwareSampler(max_nqubits=16, range_ncores=[2,8]),
    dropout=0.0,
  )
  allocator.resume_training(name, train_cfg)



if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Train qubit allocator")

  parser.add_argument("-t", "--train", action="store_true", help="Train a model from scratch")
  parser.add_argument("-f", "--finetune", action="store_true", help="Finetune an existing model")
  parser.add_argument("-b", "--benchmark", action="store_true", help="Benchmark existing model")
  parser.add_argument("-s", "--comp_sota", action="store_true", help="Compare with sota")
  parser.add_argument("-n", "--name", type=str, help="Path to the model file")
  parser.add_argument("-c", "--checkpoint", type=int, help="Checkpoint number to load")
  parser.add_argument("-d", "--data_dir", type=str, help="Data folder directory")
  args = parser.parse_args()

  ''' Train the base models with direct allocation '''
  if args.train:
    allocator = DirectAllocator(
      device='cuda',
      model_cfg=ModelConfigs(embed_size=64, num_heads=2, num_layers=2),
      mode=DirectAllocator.Mode.Sequential,
    )
    train_model_da(allocator, name=args.name)

  ''' Refine a direct allocator model '''
  if args.finetune:
    finetune_model_da(name=args.name, chkpt=args.checkpoint)

  ''' Benchmark '''
  if args.benchmark:
    validate(model_name=args.name, chkpt=args.checkpoint)
    benchmark(model_name=args.name, chkpt=args.checkpoint)
  
  ''' Compare with SOTA '''
  if args.comp_sota:
    compare_w_sota(model_name=args.name, chkpt=args.checkpoint, data_dir=args.data_dir)