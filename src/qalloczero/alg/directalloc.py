import os
import json
import torch
import gc
import warnings
from enum import Enum
from time import time
from typing import Self, Tuple, Optional, Any
from copy import deepcopy
from dataclasses import dataclass, asdict
from src.utils.conflict_handler import AGENT_HANDLER_REGISTRY, NoHandler
from src.utils.customtypes import Circuit, Hardware
from src.utils.allocutils import sol_cost, get_all_checkpoints
from scipy.stats import ttest_ind
from src.utils.mapping import map_agent_to_qubit
from src.utils.timer import Timer
from src.utils.memory import get_ram_usage
from src.sampler.hardwaresampler import HardwareSampler
from src.sampler.circuitsampler import CircuitSampler
from src.qalloczero.alg.ts import ModelConfigs
from src.qalloczero.models.predmodel import PredictionModel, PredictionModel
from src.utils.environment import QubitAllocationEnvironment, ENV_REGISTRY
from src.utils.other_utils import gather_by_index



@dataclass
class DAConfig:
  noise: float = 0.0
  mask_invalid: bool = True
  greedy: bool = True


class DirectAllocator:

  class Mode(Enum):
    Sequential = 0
    Parallel   = 1
    Fast       = 2

  @dataclass
  class TrainConfig:
    train_iters: int
    batch_size: int
    group_size: int
    validate_each: int
    validation_hardware: Hardware
    validation_circuits: list[Circuit]
    store_path: str
    initial_noise: float
    noise_decrease_factor: int
    min_noise: float
    circ_sampler: CircuitSampler
    lr: float
    inv_mov_penalization: float
    hardware_sampler: HardwareSampler
    mask_invalid: bool
    dropout: float = 0.0
    use_init_logp: bool = True
    wandb_enable: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_tags: Optional[list[str]] = None
    wandb_notes: Optional[str] = None


  def __init__(
    self,
    device: str = "cpu",
    model_cfg: ModelConfigs = ModelConfigs(),
    env: str = 'qa',
        ):
    self.model_cfg = model_cfg
    self.pred_model = PredictionModel(
      embed_size=model_cfg.embed_size,
      circuit_embds_kwargs=model_cfg.circuit_embds_kwargs,
      context_embds_kwargs=model_cfg.context_embds_kwargs,
    )
    self.pred_model.to(device)
    self.env = ENV_REGISTRY[env]()

    # TODO: Change to none and pass args
    agent_handler_type = model_cfg.conflict_handler_kwargs.pop("type", "highprob")
    agent_handler = AGENT_HANDLER_REGISTRY.get(agent_handler_type, NoHandler)
    self.conflict_handler = agent_handler(**model_cfg.conflict_handler_kwargs)
    
  

  @property
  def device(self) -> torch.device:
    return next(self.pred_model.parameters()).device


  def _save_model_cfg(self, path: str):
    params = dict(
      embed_size=self.model_cfg.embed_size,
      circuit_embds_kwargs=self.model_cfg.circuit_embds_kwargs,
      context_embds_kwargs=self.model_cfg.context_embds_kwargs,
    )
    with open(os.path.join(path, "optimizer_conf.json"), "w") as f:
      json.dump(params, f, indent=2)


  def _make_save_dir(self, path: str, overwrite: bool) -> str:
    old_path = path
    if os.path.isdir(path):
      if not overwrite:
        i = 2
        while os.path.isdir(path + f"_v{i}"):
          i += 1
        path += f"_v{i}"
        os.makedirs(path)
        warnings.warn(f"Provided folder \"{old_path}\" already exists, saving as \"{path}\"")
      else:
        warnings.warn(f"Provided folder \"{old_path}\" already exists, overwriting previous save file")
    else:
      os.makedirs(path)
    self._save_model_cfg(path)
    return path


  def _init_wandb(
    self,
    train_cfg: TrainConfig,
    save_path: str,
    wandb_config: dict[str, Any],
  ):
    if not train_cfg.wandb_enable:
      return None
    try:
      import wandb
    except ImportError as exc:
      raise ImportError(
        "Weights & Biases logging is enabled but the 'wandb' package is missing. "
        "Install it with 'pip install wandb' to proceed."
      ) from exc
    run_name = train_cfg.wandb_run_name or os.path.basename(save_path.rstrip(os.sep))
    wandb.init(
      project=train_cfg.wandb_project or "direct-allocator",
      entity=train_cfg.wandb_entity,
      name=run_name,
      group=train_cfg.wandb_group,
      tags=train_cfg.wandb_tags,
      notes=train_cfg.wandb_notes,
      config=wandb_config,
      dir=save_path,
    )
    wandb.define_metric("iter")
    wandb.define_metric("*", step_metric="iter")
    return wandb


  def save(self, path: str, overwrite: bool = False):
    path = self._make_save_dir(path=path, overwrite=overwrite)
    torch.save(self.pred_model.state_dict(), os.path.join(path, "pred_mod.pt"))
    return path


  @staticmethod
  def load(path: str, device: str = "cuda", checkpoint: Optional[int] = None) -> Self:
    if not os.path.isdir(path):
      raise Exception(f"Provided load directory does not exist: {path}")
    with open(os.path.join(path, "optimizer_conf.json"), "r") as f:
      params = json.load(f)
    model_cfg = ModelConfigs(
      embed_size=params['embed_size'],
      num_heads=params['num_heads'],
      num_layers=params['num_layers'],
    )
    loaded = DirectAllocator(device=device, model_cfg=model_cfg)
    model_file = "pred_mod.pt"
    if checkpoint is not None:
      chpt_files = get_all_checkpoints(path)
      if checkpoint == -1:
        checkpoint = max(list(chpt_files.keys()))
      elif checkpoint not in chpt_files.keys():
        raise Exception(f'Checkpoint {checkpoint} not found: {", ".join(list(chpt_files.keys()))}')
      model_file = chpt_files[checkpoint]
    loaded.pred_model.load_state_dict(
      torch.load(
        os.path.join(path, model_file),
        weights_only=False,
        map_location=device,
      )
    )
    return loaded
  

  def _allocate(
    self,
    allocations: torch.Tensor,
    circuit: Circuit,
    cfg: DAConfig,
    hardware: Hardware,
    ret_train_data: bool,
    verbose: bool = False
  ):
    device = self.device
    self.pred_model.output_logits(True)
    self.pred_model.output_demands(True) # used in the conflict handler
    self.env.reset(circuit=circuit, hardware=hardware)

    env_device = hardware.core_capacities.device

    if self.env.current_assignment is None:
      self.env.current_assignment = torch.full(
        (circuit.n_qubits,), hardware.n_cores, dtype=torch.long, device=env_device
      )

    if ret_train_data:
      all_probs: list[torch.Tensor] = []
      all_valid: list[torch.Tensor] = []

    step = 0
    # Get the embeddings of all the slices at once
    #TODO: We'll remove the batch dim for the whole pipeline
    adj_matrices = circuit.adj_matrices.to(device)
    slice_embds = self.pred_model.get_circuit_embds(adj_matrices)
    
    while not self.env.finished:
      slice_idx = self.env.current_slice

      action_mask = self.env.get_mask().to(device)

      curr_core_allocs = self.env.current_assignment.to(device)
      
      if slice_idx == 0:
        prev_core_allocs = torch.full_like(curr_core_allocs, hardware.n_cores, device=device)
      else:
        prev_core_allocs = self.env.prev_slice_allocations.to(device)

      core_caps_vec = self.env.current_core_caps if self.env.current_core_caps is not None else hardware.core_capacities
      core_caps = core_caps_vec.to(device)

      # Retrieve the embedding for this slice 
      # TODO: Perhaps our code could be parallelized for GRPO
      # we would have to change the way we handle the env... 
      # NOTE: right now we don't have padding
      slice_embd = slice_embds[slice_idx, :, :]
      output = self.pred_model(
        slice_embd,
        prev_core_allocs=prev_core_allocs,
        current_core_allocs=curr_core_allocs,
        core_capacities=core_caps,
        core_size=hardware.core_capacities.to(device),
        core_connectivity=hardware.core_connectivity.to(device),
        adj_matrix=adj_matrices[slice_idx, :, :].to(device),
        action_mask=action_mask,
      )

      # render env (for debugging)
      #self.env.render(agent_mapping=self.pred_model.grouper.q_to_agent) 

      logits = output.logits
      final_mask = output.final_mask
      agent_demands = output.agent_demands
      probs = output.probs
      
      # Ensure every agent has at least the buffer action valid to avoid NaNs
      assert (~final_mask.any(dim=-1)).all().item() == False, \
        "Invalid final mask with no valid actions for some agents"

      pol = torch.softmax(logits, dim=-1)
      
      # Add some noise for exploration 
      if cfg.noise != 0:
        noise = torch.abs(torch.randn_like(pol))
        noise[~final_mask] = 0
        pol = (1 - cfg.noise) * pol + cfg.noise * noise
        pol = pol / pol.sum(dim=-1, keepdim=True)

      actions = pol.argmax(dim=-1) if cfg.greedy else torch.distributions.Categorical(pol).sample()
      log_pol = torch.log(pol + 1e-20)

      # TODO: Let's call conflict handler. It will return the valid actions
      # Problem: how can I use the conflicts in the reward?
      final_actions, conflict_mask, halting_ratio = self.conflict_handler(
        actions=actions,
        probs=probs,
        core_capacities=core_caps,
        agent_demands=agent_demands,
        buffer_index=hardware.n_cores
      )

      # We assume the agent only selects valid actions (guaranteed by decoder masking)
      if conflict_mask is not None:
        valid = ~conflict_mask
      else:
        valid = torch.ones_like(actions, dtype=torch.bool)

      actions_gather = actions if hasattr(cfg, 'use_init_logp') and cfg.use_init_logp else final_actions
      if ret_train_data:
        selected_log_probs = log_pol.gather(1, actions_gather.unsqueeze(1)).squeeze(1)
        all_probs.append(selected_log_probs)
        all_valid.append(valid)

      actions_q = self.pred_model.grouper.agents_to_qubits(
        final_actions, 
        current_core_allocs=curr_core_allocs, 
        num_cores=hardware.n_cores
      )

      self.env.allocate(actions_q)
      step += 1
      if verbose:
        print((f"\033[2K\r - Slice {slice_idx+1}/{circuit.n_slices} step {step}"), end="")

    allocations.copy_(self.env.allocations)
    if verbose:
      print('\033[2K\r', end='')
    if ret_train_data:
      return torch.cat(all_probs), torch.cat(all_valid), None
    


  def optimize(
    self,
    circuit: Circuit,
    hardware: Hardware,
    cfg: DAConfig = DAConfig(),
    verbose: bool = False
  ) -> Tuple[torch.Tensor, float]:
    if circuit.n_qubits != hardware.n_qubits:
      raise Exception((
        f"Number of physical qubits does not match number of qubits in the "
        f"circuit: {hardware.n_qubits} != {circuit.n_qubits}"
      ))
    self.pred_model.eval()
    allocations = torch.empty([circuit.n_slices, circuit.n_qubits], dtype=torch.int)
    self._allocate(
      allocations=allocations,
      circuit=circuit,
      cfg=cfg,
      hardware=hardware,
      ret_train_data=False,
      verbose=verbose,
    )
    cost = sol_cost(allocations=allocations, core_con=hardware.core_connectivity)
    return allocations, cost


  def _update_best(
    self,
    val_cost: torch.Tensor,
    save_path:str,
    it: int,
  ):
    vc_mean=val_cost.mean().item()
    chkpt_name = f"checkpt_{it+1}_{int(vc_mean*1000)}.pt"
    torch.save(self.pred_model.state_dict(), os.path.join(save_path, chkpt_name))
    best_model = dict(
      val_cost=val_cost,
      vc_mean=vc_mean,
    )
    print(f"saving as {chkpt_name}")
    return best_model


  def train(
    self,
    train_cfg: TrainConfig,
  ) -> dict[str, list]:
    self.iter_timer = Timer.get("_train_iter_timer")
    self.iter_timer.reset()
    optimizer = torch.optim.Adam(self.pred_model.parameters(), lr=train_cfg.lr)
    opt_cfg = DAConfig(
      noise=train_cfg.initial_noise,
      mask_invalid=train_cfg.mask_invalid,
      greedy=False,
    )
    opt_cfg.use_init_logp = train_cfg.use_init_logp
    data_log = dict(
      train_cfg = dict(
        train_iters=train_cfg.train_iters,
        batch_size=train_cfg.batch_size,
        group_size=train_cfg.group_size,
        validate_each=train_cfg.validate_each,
        initial_noise=train_cfg.initial_noise,
        noise_decrease_factor=train_cfg.noise_decrease_factor,
        lr=train_cfg.lr,
        inv_mov_penalization=train_cfg.inv_mov_penalization,
        hws_nqubits=train_cfg.hardware_sampler.max_nqubits,
        hws_range_ncores=train_cfg.hardware_sampler.range_ncores,
        min_noise=train_cfg.min_noise,
        mask_invalid=train_cfg.mask_invalid,
        dropout=train_cfg.dropout,
        use_init_logp=train_cfg.use_init_logp,
        allocator=str(train_cfg.circ_sampler),
        wandb_enable=train_cfg.wandb_enable,
        wandb_project=train_cfg.wandb_project,
        wandb_entity=train_cfg.wandb_entity,
        wandb_run_name=train_cfg.wandb_run_name,
        wandb_group=train_cfg.wandb_group,
        wandb_tags=train_cfg.wandb_tags,
        wandb_notes=train_cfg.wandb_notes,
      ),
      advantage_extremes = [],
      val_cost = [],
      loss = [],
      cost_loss = [],
      val_loss = [],
      noise = [],
      vm=[],
      t = []
    )
    self.pred_model.set_dropout(train_cfg.dropout)
    init_t = time()
    best_model = dict(val_cost=None, vc_mean=None)
    save_path = self._make_save_dir(train_cfg.store_path, overwrite=False)
    wandb_logger = self._init_wandb(train_cfg, save_path, deepcopy(data_log['train_cfg']))

    try:
      for it in range(train_cfg.train_iters):
        # Train
        pheader = f"\033[2K\r[{it + 1}/{train_cfg.train_iters}]"
        self.iter_timer.start()

        loss, cost_loss, val_loss, vm_ratio, adv_ext = self._train_batch(
          pheader=pheader,
          optimizer=optimizer,
          opt_cfg=opt_cfg,
          train_cfg=train_cfg,
        )

        # Validate
        if (it+1)%train_cfg.validate_each == 0:
          print(f"\033[2K\r      Running validation...", end='')
          with torch.no_grad():
            val_cost = self._validation(train_cfg=train_cfg)
          vc_mean = val_cost.mean().item()
          data_log['val_cost'].append(vc_mean)
          print(f"\033[2K\r      vc={vc_mean:.4f}, ", end='')
          val_std = val_cost.std(unbiased=False).item()
          if wandb_logger is not None:
            wandb_logger.log(
              {
                "iter": it + 1,
                "val_cost": vc_mean,
                "val_cost_std": val_std,
              },
              step=it + 1,
            )
          if best_model['val_cost'] is None:
            best_model = self._update_best(val_cost, save_path, it)
          else:
            p = ttest_ind(val_cost.numpy(), best_model['val_cost'].numpy(), equal_var=False)[1]
            if p < 0.2:
              if vc_mean < best_model['vc_mean']:
                print(f"better than prev {best_model['vc_mean']:.4f} with p={p:.3f}, updating and ", end='')
                best_model = self._update_best(val_cost, save_path, it)
              else:
                print(f"worse than prev {best_model['vc_mean']:.4f} with p={p:.3f}, ", end='')
                self._update_best(val_cost, save_path, it)
            else:
              print(f"not enough significance wrt prev={best_model['vc_mean']:.4f} p={p:.3f}, ", end='')
              self._update_best(val_cost, save_path, it)
          with open(os.path.join(save_path, "train_data.json"), "w") as f:
            json.dump(data_log, f, indent=2)

        self.iter_timer.stop()
        t_left = self.iter_timer.avg_time * (train_cfg.train_iters - it - 1)

        print((
          f"{pheader} l={loss:.3f} (c={cost_loss:.3f} v={val_loss:.3f}) \t n={opt_cfg.noise:.3f} "
          f"vm={vm_ratio:.3f} t={self.iter_timer.time:.2f}s "
          f"({int(t_left)//3600:02d}:{(int(t_left)%3600)//60:02d}:{int(t_left)%60:02d} est. left) "
          f"ram={get_ram_usage():.2f}GB"
        ))
        
        data_log['loss'].append(loss)
        data_log['cost_loss'].append(cost_loss)
        data_log['val_loss'].append(val_loss)
        data_log['noise'].append(opt_cfg.noise)
        elapsed = time() - init_t
        data_log['t'].append(elapsed)
        data_log['vm'].append(vm_ratio)
        data_log['advantage_extremes'].append(adv_ext)
        adv_min = adv_max = None
        if isinstance(adv_ext, list) and len(adv_ext) > 0:
          adv_min = min(v[0] for v in adv_ext)
          adv_max = max(v[1] for v in adv_ext)
        elif isinstance(adv_ext, tuple):
          adv_min, adv_max = adv_ext
        if wandb_logger is not None:
          wandb_payload = dict(
            iter=it + 1,
            loss=loss,
            cost_loss=cost_loss,
            val_loss=val_loss,
            noise=opt_cfg.noise,
            valid_move_ratio=vm_ratio,
            elapsed_s=elapsed,
          )
          if adv_min is not None and adv_max is not None:
            wandb_payload['advantage_min'] = adv_min
            wandb_payload['advantage_max'] = adv_max
          wandb_logger.log(wandb_payload, step=it + 1, commit=False)
        opt_cfg.noise = max(train_cfg.min_noise, opt_cfg.noise*train_cfg.noise_decrease_factor)

    except KeyboardInterrupt as e:
      if 'y' not in input('\nGraceful shutdown? [y/n]: ').lower():
        raise e
    finally:
      if wandb_logger is not None:
        wandb_logger.finish()
    torch.save(self.pred_model.state_dict(), os.path.join(save_path, "pred_mod.pt"))
    with open(os.path.join(save_path, "train_data.json"), "w") as f:
      json.dump(data_log, f, indent=2)
  

  def _train_batch(
    self,
    pheader: str,
    optimizer: torch.optim.Optimizer,
    opt_cfg: DAConfig,
    train_cfg: TrainConfig,
  ) -> float:
    self.pred_model.train()
    n_total = train_cfg.batch_size*train_cfg.group_size
    inv_pen = train_cfg.inv_mov_penalization
    advantage_extremes = []

    while True:
      total_loss = 0
      total_cost_loss = 0
      total_valid_loss = 0
      valid_moves_ratio = 0
      optimizer.zero_grad()
      try:
        # with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
        for batch_i in range(train_cfg.batch_size):
          hardware = train_cfg.hardware_sampler.sample()
          train_cfg.circ_sampler.num_lq = hardware.n_qubits
          circuit = train_cfg.circ_sampler.sample()
          all_costs = torch.empty([train_cfg.group_size], device=self.device)
          action_log_probs = torch.empty([train_cfg.group_size], device=self.device)
          inv_moves_sum = torch.empty([train_cfg.group_size], device=self.device)

          #TODO: parallelize this?
          for group_i in range(train_cfg.group_size):
            opt_n = group_i + batch_i*train_cfg.group_size
            print(f"{pheader} ns={circuit.n_slices} nq={hardware.n_qubits} nc={hardware.n_cores} Optimizing {opt_n + 1}/{n_total}", end='')
            allocations = torch.empty([circuit.n_slices, circuit.n_qubits], dtype=torch.int)
            log_probs, valid_moves, unalloc_probs = self._allocate(
              allocations=allocations,
              circuit=circuit,
              cfg=opt_cfg,
              hardware=hardware,
              ret_train_data=True,
            )
            cost = sol_cost(allocations=allocations.detach(), core_con=hardware.core_connectivity)
            all_costs[group_i] = cost/(circuit.n_gates_norm + 1)
            action_log_probs[group_i] = torch.sum(log_probs[valid_moves.detach()])
            if unalloc_probs is not None:
              action_log_probs[group_i] += torch.sum(torch.log(unalloc_probs))
            inv_moves_sum[group_i] = torch.sum(log_probs[~valid_moves.detach()])
            valid_moves_ratio += valid_moves.float().mean().item()

          all_costs = (all_costs - all_costs.mean()) / (all_costs.std(unbiased=True) + 1e-8)
          advantage_extremes.append((all_costs.min().item(), all_costs.max().item()))
          n_samps = (train_cfg.batch_size * circuit.n_steps)
          cost_loss = (1 - inv_pen) * torch.sum(all_costs*action_log_probs) / n_samps
          total_cost_loss += cost_loss.item()
          valid_loss = inv_pen * torch.sum(inv_moves_sum) / n_samps
          total_valid_loss += valid_loss.item()
          loss = cost_loss + valid_loss
          loss.backward()
          total_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(self.pred_model.parameters(), max_norm=1)
        optimizer.step()
        break
      except torch.cuda.OutOfMemoryError:
        print(" Ran out of VRAM! Retrying...")
        if 'loss' in locals(): del loss
        if 'cost_loss' in locals(): del cost_loss
        if 'valid_loss' in locals(): del valid_loss
        if 'action_log_probs' in locals(): del action_log_probs
        if 'inv_moves_sum' in locals(): del inv_moves_sum
        if 'log_probs' in locals(): del log_probs
        if 'allocations' in locals(): del allocations
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return (
      total_loss,
      total_cost_loss,
      total_valid_loss,
      valid_moves_ratio/(train_cfg.batch_size*train_cfg.group_size),
      advantage_extremes[0] if train_cfg.batch_size == 1 else advantage_extremes,
    )
  

  def _validation(self, train_cfg: TrainConfig) -> float:
    da_cfg = DAConfig()
    norm_costs = torch.empty([len(train_cfg.validation_circuits)])
    for i, circ in enumerate(train_cfg.validation_circuits):
      norm_costs[i] = self.optimize(circ, cfg=da_cfg, hardware=train_cfg.validation_hardware)[1]/(circ.n_gates_norm + 1)
    return norm_costs