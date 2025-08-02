from typing import Union
import torch
from torch.nn import Module


class InferenceServer:
  ''' Handles the execution (inference) of all different models.
  '''
  REPR_MODEL: Union[Module, None] = None
  PRED_MODEL: Union[Module, None] = None
  DYN_MODEL:  Union[Module, None] = None


  @staticmethod
  def addRepresentationModel(model: Module):
    ''' This model will be in charge of encoding circuits.

    It should take a tensor like circuit_slice_matrices and return an encoded circuit.
    '''
    InferenceServer.REPR_MODEL = model
  

  @staticmethod
  def addPredModel(model: Module):
    ''' This model will be in charge of, given the current state and capacities, predict the policy
    (core assignation probability priors) and value for the current state.
    '''
    InferenceServer.PRED_MODEL = model
  

  @staticmethod
  def addDynamicsModel(model: Module):
    ''' This model will be in charge of obtaining the next state given the current state and an
    action (core allocation).
    '''
    InferenceServer.DYN_MODEL = model