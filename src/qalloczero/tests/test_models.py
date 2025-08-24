from typing import Tuple
import torch


class SimpleModelLarge(torch.nn.Module):
  CREATED = False

  def __init__(self):
    super().__init__()
    if SimpleModelLarge.CREATED:
      print("YOU ARE DUPLICATING A MODEL")
    else:
      SimpleModelLarge.CREATED = True
    self.fc1 = torch.nn.Sequential(
      torch.nn.Linear(256,256),
      torch.nn.ReLU(),
      torch.nn.Linear(256,64),
      torch.nn.ReLU()
    )
    self.fc2 = torch.nn.Sequential(
      torch.nn.Linear(512,512),
      torch.nn.ReLU(),
      torch.nn.Linear(512,128+64),
      torch.nn.ReLU()
    )
    self.fc = torch.nn.Sequential(
      torch.nn.Linear(256,256),
      torch.nn.ReLU(),
      torch.nn.Linear(256,128),
      torch.nn.ReLU(),
      torch.nn.Linear(128,64),
      torch.nn.ReLU()
    )
    self.fc_v = torch.nn.Linear(64,1)
  
  @staticmethod
  def input_sizes() -> Tuple[int, int]:
    return 256, 512
  
  def forward(self, x0: torch.Tensor, x1: torch.Tensor, flag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    x0 = self.fc1(x0)
    x0[flag.T.squeeze(0),:] = 1 - x0[flag.T.squeeze(0),:]
    x1 = self.fc2(x1)
    x = torch.concat([x0, x1], axis=-1)
    y0 = self.fc(x)
    y1 = self.fc_v(y0)
    return y0, y1
  
  def unbatched_forward(self, x0: torch.Tensor, x1: torch.Tensor, flag: torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    y0, y1 = self(x0=x0.unsqueeze(0), x1=x1.unsqueeze(0), flag=flag.unsqueeze(0))
    return y0.squeeze(0), y1.squeeze(0)



class SimpleModelSmall(torch.nn.Module):
  CREATED = False

  def __init__(self):
    super().__init__()
    if SimpleModelSmall.CREATED:
      print("YOU ARE DUPLICATING A MODEL")
    else:
      SimpleModelSmall.CREATED = True
    self.fc1 = torch.nn.Sequential(
      torch.nn.Linear(32,16),
      torch.nn.ReLU(),
      torch.nn.Linear(16,12),
      torch.nn.ReLU()
    )
    self.fc2 = torch.nn.Sequential(
      torch.nn.Linear(16,8),
      torch.nn.ReLU(),
      torch.nn.Linear(8,4),
      torch.nn.ReLU()
    )
    self.fc = torch.nn.Sequential(
      torch.nn.Linear(16,8),
      torch.nn.ReLU(),
      torch.nn.Linear(8,4),
      torch.nn.ReLU()
    )
    self.fc_v = torch.nn.Linear(4,1)
  
  @staticmethod
  def input_sizes() -> Tuple[int, int]:
    return 32, 16
  
  def forward(self, x0: torch.Tensor, x1: torch.Tensor, flag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    x0 = self.fc1(x0)
    x0[flag.T.squeeze(0),:] = 1 - x0[flag.T.squeeze(0),:]
    x1 = self.fc2(x1)
    x = torch.concat([x0, x1], axis=-1)
    y0 = self.fc(x)
    y1 = self.fc_v(y0)
    return y0, y1
  
  def unbatched_forward(self, x0: torch.Tensor, x1: torch.Tensor, flag: torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    y0, y1 = self(x0=x0.unsqueeze(0), x1=x1.unsqueeze(0), flag=flag.unsqueeze(0))
    return y0.squeeze(0), y1.squeeze(0)
