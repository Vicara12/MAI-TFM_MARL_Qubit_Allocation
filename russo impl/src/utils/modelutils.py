import os
import json
from typing import Dict, Union
import torch
from os.path import dirname, join
from datetime import datetime
from alg.qubitallocator import QubitAllocator

def getTrainFolderPath():
  return join(dirname(dirname(dirname(os.path.abspath(__file__)))), "trained")


def getFolderName(num_lq: int, sampler: str):
  return f"{sampler}_{num_lq}nlq_" + datetime.now().strftime("%y%m%d_%H%M%S")


def storeTrain(allocator: QubitAllocator, train_data: Dict, folder_name: Union[str, None]):
  path = getTrainFolderPath()
  if folder_name is None:
    folder = getFolderName()
  with open(join(path, folder, "train_params.json"), "w") as f:
    json.dump(train_data, f, indent=2)
  torch.save(allocator, join(path, folder, "model.pth"))


def loadModel(folder_name: str):
  path = getTrainFolderPath()
  model = torch.load(join(path, folder_name, "model.pth"))
  with open(join(path, folder_name, "train_params.json"), "r") as f:
    train_data = json.loads(f.read())
  return model, train_data