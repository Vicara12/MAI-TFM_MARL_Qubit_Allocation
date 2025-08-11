from typing import List, Self, Tuple, Dict, Optional
from math import sqrt, log
from copy import copy
from dataclasses import dataclass
import torch
from utils.customtypes import Hardware, Circuit, GateType
from qalloczero.models.inferenceserver import InferenceServer



class MCTS:
  ''' A class that performs a round of Monte Carlo Tree Search driven by DL heuristics.

  Args:
    - init_repr: initial representation of the state (s_0 in Ref. [1]).

  References:
    [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm]
    (https://arxiv.org/abs/1712.01815)
      David Silver et. al. 2017.
  '''

  @dataclass
  class Node:
    # Circuit sate attributes
    current_allocs: torch.Tensor = None
    prev_allocs: torch.Tensor = None # Allocations in the previous time slice
    prev_core_embs: torch.Tensor = None
    core_caps: torch.Tensor = None
    # State attributes
    allocation_step: int = None
    terminal: bool = False
    current_slice: int = None
    # RL attributes
    policy: torch.Tensor = None
    value_sum: float = None
    visit_count: int = 0
    cost:      int = None
    children: Dict[int, Self] = None

    @property
    def expanded(self) -> bool:
      return self.children is not None

    @property
    def value(self) -> float:
      return self.value_sum/self.visit_count if self.visit_count != 0 else 0



  @dataclass
  class Config:
    noise: float = 0.25
    dirichlet_alpha: float = 0.3
    discount_factor: float = 1.0
    ucb_c1: float = 1.25  # As in Ref. [1]
    ucb_c2: float = 19652 # As in Ref. [1]



  def __init__(self,
               slice_embs: torch.Tensor,
               circuit_embs: torch.Tensor,
               circuit: Circuit,
               hardware: Hardware,
               config: Config):
    assert InferenceServer.hasModel("pred_model"), "No prediction model found in the InferenceServer"
    assert InferenceServer.hasModel("snap_enc_model"), "No snapshot encoder model found in the InferenceServer"
    self.slice_embs = slice_embs
    self.circuit_embs = circuit_embs
    self.circuit = circuit
    self.hardware = hardware
    self.cfg = config
    self.root = self.__buildRoot()


  def iterate(self, target_tree_size: int) -> Tuple[int, int]:
    # Visit count is equal to the current size of the tree (excluding non-expanded nodes)
    num_sims = target_tree_size - self.root.visit_count
    for _ in range(num_sims):
      node = self.root
      search_path = [node]
      while node.expanded and not node.terminal:
        node = self.__selectChild(current_node=node)
        search_path.append(node)
      if not node.terminal:
        self._expandNode(node)
      self.__backprop(search_path)
    action = self.__selectAction(self.root)
    self.root = self.root.children[action]
    return action, num_sims


  @staticmethod
  def __selectAction(node: Node) -> int:
    visit_counts = torch.tensor(list(child.visit_count for child in node.children.values()),
                                dtype=torch.float)
    # Return a random core allocation with probability proportional to the visit count of each of
    # the node's children
    action_idx = torch.multinomial(visit_counts, num_samples=1).item()
    return list(node.children.keys())[action_idx]


  def __getNewPolicyAndValue(self, node: Node) -> Tuple[torch.Tensor, torch.Tensor]:
    if node.terminal:
      return None, 0
    pol, v = InferenceServer.inference(
      model_name="pred_model",
      current_alloc=self.circuit.alloc_steps[node.allocation_step],
      core_embs=node.prev_core_embs,
      prev_core_allocs=node.prev_allocs,
      current_core_capacities=node.core_caps,
      circuit_emb=self.circuit_embs[node.current_slice],
      slice_emb=self.slice_embs[node.current_slice],
    )
    # Add exploration noise to the priors
    dir_noise = torch.distributions.Dirichlet(self.cfg.dirichlet_alpha * torch.ones_like(pol)).sample()
    pol = (1 - self.cfg.noise)*pol + self.cfg.noise*dir_noise
    # Set prior of cores that do not have space for this alloc to zero
    n_qubits = len(self.circuit.alloc_steps[node.allocation_step][1])
    pol[node.core_caps < n_qubits] = 0
    pol /= sum(pol)
    return pol, v
  

  def __getCoreEmb(self, core_allocs: torch.Tensor) -> torch.Tensor:
    return InferenceServer.inference(model_name="snap_enc_model", core_allocs=[core_allocs])
  

  def __buildRoot(self) -> Node:
    root = MCTS.Node(
      current_allocs = -1*torch.ones(size=(self.circuit.n_qubits,), dtype=int),
      prev_allocs = None,
      prev_core_embs = self.__getCoreEmb(-1*torch.ones(size=(self.circuit.n_qubits,), dtype=int)),
      core_caps = self.hardware.core_capacities.clone(),
      allocation_step = 0,
      current_slice = 0,
      cost = 0
    )
    root.policy, root.value_sum = self.__getNewPolicyAndValue(root)
    return root


  def _expandNode(self, node: Node):
    if node.terminal:
      return
    node.children = {}
    _, qubits_to_alloc = self.circuit.alloc_steps[node.allocation_step]
    slice_idx_children, _ = self.circuit.alloc_steps[node.allocation_step+1]

    for action in range(self.hardware.n_cores):
      if node.policy[action] == 0:
        continue
      child = MCTS.Node()
      node.children[action] = child

      if slice_idx_children != node.current_slice:
        child.current_allocs = -1*torch.ones_like(node.current_allocs)
        child.prev_allocs = node.current_allocs.clone()
        child.prev_allocs[qubits_to_alloc,] = action
        child.prev_core_embs = self.__getCoreEmb(child.prev_allocs)
        child.core_caps = self.hardware.core_capacities.clone()
      else:
        child.current_allocs = node.current_allocs.clone()
        child.current_allocs[qubits_to_alloc,] = action
        child.prev_allocs = node.prev_allocs
        child.prev_core_embs = node.prev_core_embs
        child.core_caps = node.core_caps.clone()
        child.core_caps[action] -= len(qubits_to_alloc)
        assert child.core_caps[action] >= 0, 'Not enough space in core to expand'
      child.allocation_step = node.allocation_step+1
      child.terminal = (child.allocation_step == len(self.circuit.alloc_steps)-1)
      child.current_slice = slice_idx_children
      child.cost = self.__computeActionCost(node, action)

      child.policy, child.value_sum = self.__getNewPolicyAndValue(child)
  

  def __computeActionCost(self, node: Node, action: int) -> int:
    if node.current_slice == 0:
      return 0
    _, qubits_to_alloc = self.circuit.alloc_steps[node.allocation_step]
    prev_cores = node.prev_allocs[qubits_to_alloc,]
    costs = self.hardware.core_connectivity[action, prev_cores]
    return torch.sum(costs).item()


  def __backprop(self, search_path: List[Node]):
    search_path[-1].visit_count += 1
    # Reverse list order and pair items. For example [0,1,2,3] -> ((2,3), (1,2), (0,1))
    for node, next_node in zip(search_path[-2::-1], search_path[:0:-1]):
      node.value_sum += next_node.value + next_node.cost
      node.visit_count += 1
  
  
  def __UCB(self, node: Node, action: int) -> float:
    ''' Upper Confidence Bound with minus sign in CB to account for minimization of cost.

    For a nicely formatted version of this formula refer to Appendix B in Ref. [1]
    '''
    return (node.children[action].value + node.children[action].cost) - \
           node.policy[action]*sqrt(node.visit_count)/(1+node.children[action].visit_count) * \
              (self.cfg.ucb_c1 + log((node.visit_count + self.cfg.ucb_c2 + 1)/self.cfg.ucb_c2))

  
  def __selectChild(self, current_node: Node) -> Tuple[int, Node]:
    (_, action) = min((self.__UCB(current_node, a), a) for a in current_node.children.keys())
    return current_node.children[action]