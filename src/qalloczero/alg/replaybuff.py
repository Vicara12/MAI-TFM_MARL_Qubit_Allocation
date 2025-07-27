from typing import Self, Tuple
import torch
import warnings

class ReplayBuffer(object):
  """Efficient version of ReplayBuffer using circular buffers on torch tensors."""
  def __init__(self, buffer_size: int, state_len: int) -> Self:
    self.buff_size = buffer_size
    self.n_elms = 0
    self.idx = 0
    self.read_idx = 0
    self.full = False
    self.sampling_mode = False
    self.state   = torch.empty(size=(buffer_size, state_len), dtype=torch.float)
    self.actions = torch.empty(size=(buffer_size, 1),         dtype=torch.float)
    self.rewards = torch.empty(size=(buffer_size, 1),         dtype=torch.float)

  def __len__(self):
    return self.n_elms
  
  def __circularPush(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor) -> None:
    ''' Push all elements in src into dst with circular buffer behavior.

    The types of both matrices should coincide and src.shape[1] == dst.shape[1]. It is also assumed
    that src.shape[0] < dst.shape[0].
    '''
    n_elms = states.shape[0]
    # Not enough elements to wrap
    if self.buff_size >= (self.idx + n_elms):
      self.state[self.idx:self.idx + n_elms, :].copy_(states)
      self.actions[self.idx:self.idx + n_elms, :].copy_(actions)
      self.rewards[self.idx:self.idx + n_elms, :].copy_(rewards)
    # Wrap to start
    else:
      split = self.buff_size - self.idx
      self.state[-split:,:].copy_(states[:split,:])
      self.actions[-split:,:].copy_(actions[:split,:])
      self.rewards[-split:,:].copy_(rewards[:split,:])
      self.state[:(n_elms-split),:].copy_(states[:(n_elms-split),:])
      self.actions[:(n_elms-split),:].copy_(actions[:(n_elms-split),:])
      self.rewards[:(n_elms-split),:].copy_(rewards[:(n_elms-split),:])
  
  def __circularSample(self, batch_size, device):
    # Not enough elements to wrap
    if self.n_elms >= (self.read_idx + batch_size):
      sample_lambda = lambda t: torch.as_tensor(t[self.read_idx:self.read_idx + batch_size, :], device=device)
    # Wrap to start
    else:
      split = self.n_elms - self.read_idx
      sample_lambda = lambda t: torch.as_tensor(torch.concat([t[self.read_idx:self.n_elms,:],
                                                              t[:(self.n_elms-split),:]], axis=0), device=device)
    return sample_lambda(self.state), sample_lambda(self.actions), sample_lambda(self.rewards)

    
  def setSamplingMode(self):
    ''' Set replay buffer in sample mode.
    '''
    self.read_idx = 0
    self.sampling_mode = True
    # Shuffle tensors
    shuffled_indices = torch.randperm(self.n_elms)
    self.state[:self.n_elms]   = self.state[shuffled_indices]
    self.actions[:self.n_elms] = self.actions[shuffled_indices]
    self.rewards[:self.n_elms] = self.rewards[shuffled_indices]

  def setPushMode(self):
    ''' Set replay buffer in push mode (clears all values).
    '''
    self.n_elms = 0
    self.idx = 0
    self.full = False
    self.sampling_mode = False

  def push(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor) -> None:
    ''' Insert a tensor of values into the buffer.
    '''
    if self.sampling_mode:
      raise Exception("tried to push into a replay buffer in sampling mode")
    self.__circularPush(states, actions, rewards)
    self.full = (self.full or (self.idx + states.shape[0] > self.buff_size))
    self.idx = (self.idx + states.shape[0]) % self.buff_size
    self.n_elms = min(self.n_elms + states.shape[0], self.buff_size)

  def sample(self, batch_size: int, device: str, warn: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not self.sampling_mode:
      raise Exception("tried to sample from a replay buffer in push mode")
    states, actions, rewards = self.__circularSample(batch_size, device)
    if warn and self.n_elms < self.read_idx + batch_size:
      warnings.warn("replay buffer sampling exceeded buffer size, wrapping", category=UserWarning)
    self.read_idx = (self.read_idx + batch_size)%self.n_elms
    return states, actions, rewards