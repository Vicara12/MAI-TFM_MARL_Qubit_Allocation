from typing import Tuple
import torch
from utils.customtypes import Hardware


class PredictionModel(torch.nn.Module):
  def __init__(self, hardware: Hardware):
    super().__init__()
    self.hw = hardware
  

  def forward(self, *args) -> Tuple[torch.Tensor, float]:
    return torch.ones(size=(self.hw.n_cores,)), 0