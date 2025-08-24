from __future__ import annotations
from typing import Dict, Tuple, List, Optional, Union
from threading import Thread, Event, Lock
from time import time
import torch
from queue import Queue
from torch.nn import Module
from dataclasses import dataclass, field



class InferenceServer:
  ''' Handles the execution (inference) of all models.
  '''

  @dataclass
  class InferRequest:
    inputs: Dict
    result = None
    done: Event = field(default_factory=Event)


  @dataclass
  class ModelCfg:
    name: str
    model: Module
    inference_device: str
    supports_batch: bool
    # Values contains the name of the parameter and a tuple with its type and shape
    parameters: Dict[str, Tuple[torch.dtype, Tuple[int,...]]]
    # If flexible input size, input can be smaller than specified size in the last dimension
    flexible_input_size: bool = False


  INSTANCES: Dict[str, InferenceServer] = {}
  N_INSTANCES_BATCH: int = 0
  GPU_LOCK: Lock = Lock()
  

  @staticmethod
  def get(name: str) -> InferenceServer:
    if name not in InferenceServer.INSTANCES.keys():
      raise Exception(f"No model called {name} in the InferenceServer")
    return InferenceServer.INSTANCES[name]


  @staticmethod
  def hasModel(name: str) -> bool:
    return name in InferenceServer.INSTANCES.keys()


  def __init__(self,
               model_cfg: ModelCfg,
               max_batch_size: Optional[int] = None,
               batch_fill_timeout: float = 0.01
    ):
    self.name = model_cfg.name
    if model_cfg.supports_batch and InferenceServer.N_INSTANCES_BATCH >= 1:
      raise Exception("Only one InferenceServer object with batch is supported at this time")
    if model_cfg.name in InferenceServer.INSTANCES.keys():
      raise Exception(f"A model named {model_cfg.name} is already loaded in the InferenceServer")
    self.model = model_cfg.model
    self.supports_batch = model_cfg.supports_batch
    self.device = torch.device(model_cfg.inference_device)
    self.model_params = model_cfg.parameters
    if self.supports_batch:
      assert max_batch_size is not None, \
        "A max batch size must be provided if model supports batching"
      self.max_batch_size = max_batch_size
      self.batch_fill_timeout = batch_fill_timeout
      self.inferences_w_batch = [0]*(self.max_batch_size+1)
      self.gatherer_id = 0
      self.stopped_gathering = (Event(), Event())
      # Avoid gatherer 1 to go into inference mode without having gathered any data at start
      self.stopped_gathering[1-self.gatherer_id].set()
      self.thr = (Thread(target=self._workLoop, args=(0,), daemon=True),
                  Thread(target=self._workLoop, args=(1,), daemon=True))
      self.requests = Queue()
      self.finish_ev = Event()
      self.flexible_inp_size = model_cfg.flexible_input_size
      if self.flexible_inp_size:
        raise NotImplementedError("Test this first")
      self.inp_buffer = [self._initInputBuffer(), self._initInputBuffer()]
      self.data_streams = [torch.cuda.Stream(), torch.cuda.Stream()]
      self.thr[0].start()
      self.thr[1].start()
      InferenceServer.N_INSTANCES_BATCH += 1
    InferenceServer.INSTANCES[self.name] = self


  def _initInputBuffer(self) -> Dict[str, torch.Tensor]:
    input_buffer = {}
    for param_name, param_data in self.model_params.items():
      (dtype, shape) = param_data
      input_buffer[param_name] = torch.empty(size=(self.max_batch_size, *shape),
                                             dtype=dtype,
                                             device=self.device)
    return input_buffer


  def _workLoop(self, me: int) -> None:
    other = 1 - me
    while not self.finish_ev.is_set():
      if self.gatherer_id == me and self.stopped_gathering[other].is_set():
        self.stopped_gathering[other].clear()
        with torch.cuda.stream(self.data_streams[me]):
          request_list = self._gatherRequests(me)
          self.gatherer_id = other
          self.stopped_gathering[me].set()
          # Empty request list means that IS has received finish_ev
          if len(request_list) != 0:
            self._computeRequests(request_list, me)


  def _gatherRequests(self, work_thread_id: int) -> List[InferRequest]:
    request_list = []
    if self.flexible_inp_size:
      self._zeroInputBuffers(work_thread_id)
    gpu_lock = InferenceServer.GPU_LOCK
    timeout_t = None
    while (
      len(request_list) < self.max_batch_size
      and (
        timeout_t is None or time() < timeout_t or gpu_lock.locked()
      )
    ):
      timeout_t = time() + self.batch_fill_timeout
      # Prevent blocking queue
      if not self.requests.empty():
        request = self.requests.get()
        self._addRequestInputs(request.inputs, len(request_list), work_thread_id)
        request_list.append(request)
      # Manage shutdown
      if self.finish_ev.is_set():
        for request in request_list:
          request.done.set()
        return []
    return request_list


  def _addRequestInputs(self, inputs: Dict, batch_idx: int, work_thread_id: int) -> None:
    for inp_name in self.model_params.keys():
      assert inp_name in inputs.keys(), f"Model input {inp_name} not found in infer request"
      parameter = self.inp_buffer[work_thread_id][inp_name]
      if self.flexible_inp_size:
        # Fill beginning of the buffer tensor with the input
        inp_shape = inputs[inp_name].shape
        idx = (batch_idx,) + tuple(slice(0, inp_shape[0])) + inp_shape[1:]
        parameter[idx] = inputs[inp_name].to(self.device, non_blocking=True)
      else:
        parameter[batch_idx] = inputs[inp_name].to(self.device, non_blocking=True)


  def _computeRequests(self, request_list: List[InferRequest], work_thread_id: int) -> None:
    inputs = {}
    batch_size = len(request_list)
    self.inferences_w_batch[batch_size-1] += 1
    for name, value in self.inp_buffer[work_thread_id].items():
      inputs[name] = value[:batch_size]
    assert self._modelDevice().type == self.device.type, f"Model is not at target IS device {self.device}"
    
    with InferenceServer.GPU_LOCK:
      results = self.model(**inputs)
    
    if isinstance(results, tuple):
      results = tuple(result.to('cpu', non_blocking=True) for result in results)
    else:
      results = results.to('cpu', non_blocking=True)
      
    for i, request in enumerate(request_list):
      request.result = tuple(result[i] for result in tuple(results))
      request.done.set()


  def _modelDevice(self) -> str:
    return next(self.model.parameters()).device


  def infer(self, **kwargs):
    if not self.supports_batch:
      return self.model(**kwargs)
    if self.finish_ev.is_set():
      return None
    for k, v in kwargs.items():
      assert isinstance(v, torch.Tensor), f"Input {k} is not a torch.Tensor"
    request = InferenceServer.InferRequest(inputs=kwargs)
    self.requests.put(request)
    request.done.wait()
    return request.result
  

  @property
  def batch_performance(self) -> List[int]:
    return self.inferences_w_batch
  

  def __del__(self):
    del InferenceServer.INSTANCES[self.name]
    if self.supports_batch:
      self.finish_ev.set()
      while not self.requests.empty():
        request = self.requests.get()
        request.result = None
        request.done.set()
      self.thr[0].join()
      self.thr[0].join()
      InferenceServer.N_INSTANCES_BATCH -= 1