from typing import Tuple, List
from copy import deepcopy
import torch
import numpy.testing as npt
from random import choice
from threading import Thread
from utils.timer import Timer
from qalloczero.models.inferenceserver import InferenceServer
from qalloczero.tests.test_models import SimpleModelLarge, SimpleModelSmall

model_class = SimpleModelLarge

def make_inputs():
  x0_size, x1_size = model_class.input_sizes()
  return torch.randn((x0_size,)), torch.randn((x1_size,)), torch.tensor([choice([True, False])])


def compare_eq(is_y0, is_y1, y0, y1):
  npt.assert_almost_equal(is_y0.detach().numpy(), y0.detach().numpy(), decimal=5)
  npt.assert_almost_equal(is_y1.detach().numpy(), y1.detach().numpy(), decimal=5)


def work(n_reqs: int, worker_id: int, raw_model: torch.nn.Module) -> float:
  is_model = InferenceServer.get('model')
  timer = Timer(str(worker_id))
  for _ in range(n_reqs):
    x0, x1, flag = make_inputs()
    y0, y1 = raw_model.unbatched_forward(x0=x0, x1=x1, flag=flag)
    # print(f" ----- {worker_id} -----\n{x0 = }\n{x1 = }\n{flag = }\n\n{y0 = }\n{y1 = }")
    with timer:
      is_y0, is_y1 = is_model.infer(x0=x0, x1=x1, flag=flag)
    compare_eq(is_y0, is_y1, y0, y1)


def test_correctness(n_threads: int, n_reqs: int, raw_model: torch.nn.Module):
  threads = [Thread(target=work, args=(n_reqs, i, raw_model), daemon=True) for i in range(n_threads)]
  for thread in threads:
    thread.start()
  for thread in threads:
    thread.join()
  results = [Timer.get(str(i)).total_time for i in range(n_threads)]
  print(f"time_per_thread = {results}")
  print(f"mean_time = {sum(results)/len(results)}")
  is_model = InferenceServer.get('model')
  print(f"batch_performance = {is_model.batch_performance}")


def test_correctness_single(raw_model: torch.nn.Module) -> None:
  work(1, 0, raw_model)
  print("Single comparison OK!")


def main_test_is():
  x0_size, x1_size = model_class.input_sizes()
  raw_model = model_class()
  raw_model_is = deepcopy(raw_model)
  raw_model_is.to('cuda')
  InferenceServer(model_cfg=InferenceServer.ModelCfg(
      name="model",
      model=raw_model_is,
      inference_device='cuda',
      supports_batch=True,
      parameters=dict(
        x0=(torch.float32, (x0_size,)),
        x1=(torch.float32, (x1_size,)),
        flag=(torch.bool,  (1,)) 
      ),
      flexible_input_size=False
    ),
    max_batch_size=16
  )

  # test_correctness_single(raw_model)
  test_correctness(n_threads=64, n_reqs=16, raw_model=raw_model)
  # test_correctness(n_threads=64, n_reqs=128, raw_model=raw_model)
  # test_correctness(4, 8)