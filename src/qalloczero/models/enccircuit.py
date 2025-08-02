
import torch

class EncodedCircuit:
  ''' Holds the slice encodings and returns state encodings for each time slice.

  The state is encoded through a mean of the circuit slices from the current slice until the end.
  This class holds the original circuit and the state at each time slice.

  NOTE: It is possible to either a) run the model that encodes slice embeddings in train
  mode, so that we only need to execute it once (less time) but need to hold the computational graph
  for all of them until the end (more memory); or b) run the model in test mode, so that we don't
  need to hold the computational graph of each embedding (less memory) but at training need to
  re-compute the circuit embeddings in training mode (more time). We can decide on which is better
  later on depending on what is more critical, VRAM or time.
  '''

  def __init__(self, circuit_matrices: torch.Tensor, circuit_slice_embeddings: torch.Tensor):
    self.circuit_matrices = circuit_matrices
    self.circuit_embeddings = torch.empty_like(circuit_slice_embeddings)
    for t_i in range(len(circuit_slice_embeddings)):
      self.circuit_embeddings = torch.mean(circuit_slice_embeddings[t_i:,:], axis=0)
  
  def circuitEmbAtSlice(self, slice: int) -> torch.Tensor:
    return self.circuit_embeddings[slice]