from typing import Dict
import torch
from torch.nn import Module
from dataclasses import dataclass


class InferenceServer:
  ''' Handles the execution (inference) of all models.
  '''

  @dataclass
  class Model:
    model: Module
    unpack: bool



  MODELS: Dict[str, Model] = {}


  @staticmethod
  def addModel(name: str, model: Module, unpack: bool):
    if InferenceServer.hasModel(name):
      raise Exception(f"There is already a model named \"{name}\" in the InferenceServer")
    InferenceServer.MODELS[name] = InferenceServer.Model(model=model, unpack=unpack)
  

  @staticmethod
  def hasModel(name: str):
    return name in InferenceServer.MODELS.keys()


  @staticmethod
  def inference(model_name: str, *args, **kwargs):
    if not InferenceServer.hasModel(model_name):
      raise Exception(f"No model called \"{model_name}\" has been loaded in the InferenceServer")
    model_obj = InferenceServer.MODELS[model_name]
    result = model_obj.model(*args, **kwargs)
    if model_obj.unpack:
      return result[0]
    return result