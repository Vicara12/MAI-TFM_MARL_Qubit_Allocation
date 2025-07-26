import os
import json
from typing import Dict, Union
import torch
from os.path import dirname, join
from datetime import datetime
from russo.alg.qubitallocator import QubitAllocator

def getTrainFolderPath(subfolder_name: str):
  return join(dirname(dirname(dirname(os.path.abspath(__file__)))), "trained", subfolder_name)


def genTrainFolder(num_lq: int, subfolder_name: str):
  train_path = getTrainFolderPath(subfolder_name)
  fodler_name = f"{num_lq}lq_" + datetime.now().strftime("%y%m%d_%H%M%S")
  os.makedirs(join(train_path, fodler_name), exist_ok=True)
  return fodler_name


def storeTrain(allocator: QubitAllocator, train_data: Dict, subfolder_name: str, folder_name: Union[str, None]):
  path = getTrainFolderPath(subfolder_name=subfolder_name)
  if folder_name is None:
    folder = genTrainFolder(num_lq=allocator.num_lq, subfolder_name=subfolder_name)
  with open(join(path, folder, "train_params.json"), "w") as f:
    json.dump(train_data, f, indent=2)
  torch.save(allocator, join(path, folder, "model.pth"))


def loadModel(folder_name: str, subfolder_name: str):
  path = getTrainFolderPath(subfolder_name=subfolder_name)
  model = torch.load(join(path, folder_name, "model.pth"))
  with open(join(path, folder_name, "train_params.json"), "r") as f:
    train_data = json.loads(f.read())
  return model, train_data