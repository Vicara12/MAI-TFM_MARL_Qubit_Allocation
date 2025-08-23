from __future__ import annotations
from typing import Dict, Tuple, List, Optional, Union
from threading import Thread, Event, Lock
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
    supports_batch: bool
    # Values in parameter is a bool indicating wether the argument is a Tensor and, if so, its shape
    parameters: Dict[str, Tuple[bool, Union[Tuple[int, ...]]]]
    # If flexible input size, input can be smaller than specified size in the last dimension
    flexible_input_size: bool = False
    max_batch_size: Optional[int] = None


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


  def __init__(self, model_cfg: ModelCfg):
    self.name = model_cfg.name
    if model_cfg.supports_batch and InferenceServer.N_INSTANCES_BATCH >= 1:
      raise Exception("Only one InferenceServer object with batch is supported at this time")
    if model_cfg.name in InferenceServer.INSTANCES.keys():
      raise Exception(f"A model named {model_cfg.name} is already loaded in the InferenceServer")
    self.model = model_cfg.model
    self.supports_batch = model_cfg.supports_batch
    self.device = next(self.model.parameters()).device
    self.model_params = model_cfg.parameters
    if self.supports_batch:
      assert model_cfg.max_batch_size is not None, \
        "A max batch size must be provided if model supports batching"
      self.max_batch_size = model_cfg.max_batch_size
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
      self.request_buffer = []
      self.thr[0].start()
      self.thr[1].start()
      InferenceServer.N_INSTANCES_BATCH += 1
    InferenceServer.INSTANCES[self.name] = self
  

  def _initInputBuffer(self) -> Dict[str, Union[List, torch.Tensor]]:
    input_buffer = {}
    for param_name, param_data in self.model_params.items():
      # If torch tensor
      if param_data[0]:
        input_buffer[param_name] = torch.empty(size=(self.max_batch_size, *param_data[1]))
      else:
        input_buffer[param_name] = []
    return input_buffer
  

  def _zeroInputBuffers(self, work_thread_id: int) -> None:
    for inp in self.inp_buffer[work_thread_id].values():
      if isinstance(inp, torch.Tensor):
        inp.zero_()
  

  def _workLoop(self, me: int) -> None:
    other = 1 - me
    while not self.finish_ev.is_set():
      if self.gatherer_id == me and self.stopped_gathering[other].is_set():
        self.stopped_gathering[other].clear()
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
    can_add_requests = lambda: len(request_list) < self.max_batch_size and gpu_lock.locked()
    while len(request_list) == 0 or can_add_requests():
      # Prevent blocking queue
      if not self.requests.empty():
        print(f" - Got request n {len(request_list)+1}")
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
    for inp_name, inp_cfg in self.model_params.items():
      assert inp_name in inputs.keys(), \
        f"Model input {inp_name} not found in infer request"
      # If parameter is a Tensor
      if inp_cfg[0]:
        if self.flexible_inp_size:
          # Fill beginning of the buffer tensor with the input
          idx = (batch_idx,) + tuple(slice(0,s) for s in inputs[inp_name].shape)
          self.inp_buffer[work_thread_id][inp_name][idx] = inputs[inp_name]
        else:
          self.inp_buffer[work_thread_id][inp_name][batch_idx] = inputs[inp_name]
      else:
        self.inp_buffer[work_thread_id][inp_name].append(inputs[inp_name])


  def _computeRequests(self, request_list: List[InferRequest], work_thread_id: int) -> None:
    with InferenceServer.GPU_LOCK:
      inputs = {}
      batch_size = len(request_list)
      for name, value in self.inp_buffer[work_thread_id].items():
        if isinstance(value, torch.Tensor):
          inputs[name] = value[:batch_size]
        else:
          inputs[name] = value
      results = self.model(**inputs)
    for i, request in enumerate(request_list):
      request.result = tuple(result[i] for result in tuple(results))
      request.done.set()
    # Empty lists of inputs that are not tensors for next inference
    for inp, inp_data in self.model_params.items():
      if not inp_data[0]:
        self.inp_buffer[work_thread_id][inp] = []


  def toGPU(self, device) -> None:
    self.device = device
    self.model.to(device)


  def infer(self, **kwargs):
    if not self.supports_batch:
      return self.model(**kwargs)
    if self.finish_ev.is_set():
      return None
    request = InferenceServer.InferRequest(inputs=kwargs)
    self.requests.put(request)
    request.done.wait()
    return request.result
  

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