from typing import Tuple, List
import torch
import numpy.testing as npt
from random import choice
from threading import Thread
from utils.timer import Timer
from qalloczero.models.inferenceserver import InferenceServer



class SimpleModel(torch.nn.Module):
  CREATED = False

  def __init__(self):
    super().__init__()
    if SimpleModel.CREATED:
      raise Exception('nope')
    else:
      SimpleModel.CREATED = True
    self.fc1 = torch.nn.Sequential(
      torch.nn.Linear(4,4),
      torch.nn.ReLU()
    )
    self.fc2 = torch.nn.Sequential(
      torch.nn.Linear(16,12),
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
    return 4, 16
  
  def forward(self, x0: torch.Tensor, x1: torch.Tensor, flag: List[bool]) -> Tuple[torch.Tensor, torch.Tensor]:
    x0 = self.fc1(x0)
    x0[flag,:] = 1 - x0[flag,:]
    x1 = self.fc2(x1)
    x = torch.concat([x0, x1], axis=-1)
    y0 = self.fc(x)
    y1 = self.fc_v(y0)
    return y0, y1
  
  def unbatched_forward(self, x0: torch.Tensor, x1: torch.Tensor, flag: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    y0, y1 = self(x0=x0.unsqueeze(0), x1=x1.unsqueeze(0), flag=[flag])
    return y0.squeeze(0), y1.squeeze(0)

def make_inputs():
  x0_size, x1_size = SimpleModel.input_sizes()
  return torch.randn((x0_size,)), torch.randn((x1_size,)), choice([True, False])


def work(n_reqs: int, worker_id: int) -> float:
  is_model = InferenceServer.get('model')
  raw_model = is_model.model
  timer = Timer(str(worker_id))
  for _ in range(n_reqs):
    x0, x1, flag = make_inputs()
    with timer:
      is_y0, is_y1 = is_model.infer(x0=x0, x1=x1, flag=flag)
    y0, y1 = raw_model.unbatched_forward(x0=x0, x1=x1, flag=flag)
    npt.assert_almost_equal(is_y0.detach().numpy(), y0.detach().numpy())
    npt.assert_almost_equal(is_y1.detach().numpy(), y1.detach().numpy())


def test_correctness(n_threads: int, n_reqs: int):
  threads = [Thread(target=work, args=(n_reqs, i), daemon=True) for i in range(n_threads)]
  for thread in threads:
    thread.start()
  for thread in threads:
    thread.join()
  results = [Timer.get(str(i)).total_time for i in range(n_threads)]
  print(results)


def test_correctness_single() -> None:
  is_model = InferenceServer.get('model')
  raw_model = is_model.model
  x0, x1, flag = make_inputs()
  y0, y1 = raw_model.unbatched_forward(x0=x0, x1=x1, flag=flag)
  print(f"{x0 = }\n{x1 = }\n{flag = }\n\n{y0 = }\n{y1 = }")
  is_y0, is_y1 = is_model.infer(x0=x0, x1=x1, flag=flag)
  npt.assert_almost_equal(is_y0.detach().numpy(), y0.detach().numpy())
  npt.assert_almost_equal(is_y1.detach().numpy(), y1.detach().numpy())


def main_test_is():
  x0_size, x1_size = SimpleModel.input_sizes()
  InferenceServer(model_cfg=InferenceServer.ModelCfg(
      name="model",
      model=SimpleModel(),
      supports_batch=True,
      max_batch_size=4,
      parameters={'x0': (True, (x0_size,)), 'x1': (True, (x1_size,)), 'flag': (False,)},
      flexible_input_size=False
    )
  )
  # test_correctness_single()
  test_correctness(4, 32)
  # test_correctness(4, 8)


if __name__ == '__main__':
  main_test_is()