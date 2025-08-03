from typing import Union
import torch
from torch.nn import Module


class InferenceServer:
  ''' Handles the execution (inference) of all models.
  '''
  MODEL: Union[Module, None] = None


  @staticmethod
  def addModel(model: Module):
    InferenceServer.MODEL = model

  @staticmethod
  def inference(*args):
    return InferenceServer.MODEL(*args)