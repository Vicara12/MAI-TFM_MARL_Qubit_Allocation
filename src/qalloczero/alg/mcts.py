from typing import List, Self, Tuple, Dict, Optional
from math import sqrt, log
from copy import copy
from dataclasses import dataclass
import torch
from utils.customtypes import Hardware, Circuit
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
    core_caps: torch.Tensor = None
    # State attributes
    allocation_step: int = None
    terminal: bool = False
    current_slice: int = None
    # RL attributes
    policy: torch.Tensor = None
    value_sum: float = None
    visit_count: int = 1
    cost:      int = None
    children: Dict[int, Self] = None

    @property
    def expanded(self) -> bool:
      return self.children is not None

    @property
    def value(self) -> float:
      if self.terminal:
        return self.value_sum
      return self.value_sum/self.visit_count


  @dataclass
  class Config:
    noise: float = 0.25
    dirichlet_alpha: float = 0.3
    discount_factor: float = 1.0
    ucb_c1: float = 1.25  # As in Ref. [1]
    ucb_c2: float = 19652 # As in Ref. [1]



  def __init__(self, circuit_embs: torch.Tensor, circuit: Circuit, hardware: Hardware, config: Config):
    self.circuit_embs = circuit_embs
    self.circuit = circuit
    self.hardware = hardware
    self.cfg = config
    self.__buildRoot()


  def iterate(self, target_tree_size: int) -> Tuple[int, int]:
    # Visit count is equal to the current size of the tree
    num_sims = target_tree_size - self.root.visit_count
    for _ in range(num_sims):
      node = self.root
      search_path = [node]
      last_action = None
      while node.expanded and not node.terminal:
        last_action, node = self.__selectChild(current_node=node)
        search_path.append(node)
      if not node.terminal:
        node_value = self.__initNode(node, search_path[-2], last_action)
      else:
        node_value = node.value
      self.__backprop(search_path, node_value)
    action = self.__selectAction(self.root)
    self.root = self.root.children[action]
    return action, num_sims


  @staticmethod
  def __selectAction(node: Node) -> int:
    visit_counts = torch.tensor(list(child.visit_count-1 for child in node.children.values()),
                                dtype=torch.float)
    # Return a random core allocation with probability proportional to the visit count of each of
    # the node's children
    action_idx = torch.multinomial(visit_counts, num_samples=1).item()
    return list(node.children.keys())[action_idx]


  def __getNewPolicyAndValue(self,
                             slice_idx: int,
                             alloc_step: int,
                             core_caps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # If terminal node
    if alloc_step == len(self.circuit.alloc_steps):
      return None, 0
    state = self.circuit_embs[slice_idx]
    pol, v = InferenceServer.inference(state, core_caps)
    # Add exploration noise to the priors
    dir_noise = torch.distributions.Dirichlet(self.cfg.dirichlet_alpha * torch.ones_like(pol)).sample()
    pol = (1 - self.cfg.noise)*pol + self.cfg.noise*dir_noise
    # Set prior of cores that do not have space for this alloc to zero
    n_qubits = len(self.circuit.alloc_steps[alloc_step][1])
    pol[core_caps < n_qubits] = 0
    pol /= sum(pol)
    return pol, v


  def __buildRoot(self) -> Node:
    self.root = MCTS.Node(
      current_allocs=torch.empty(size=(self.circuit.n_qubits,), dtype=int),
      prev_allocs=None,
      core_caps=self.hardware.core_capacities,
      allocation_step=0,
      current_slice=0,
      children={c: None for c in range(self.hardware.n_cores)}
    )
    self.root.policy, self.root.value_sum = self.__getNewPolicyAndValue(slice_idx=0, alloc_step=0,
                                                                        core_caps=self.root.core_caps)
    self._expandNode(self.root)


  def __initNode(self,
                 node: Node,
                 parent_node: Node,
                 action: int):
    (_, alloc_qubits) = self.circuit.alloc_steps[parent_node.allocation_step]
    node.current_allocs = torch.empty_like(parent_node.current_allocs, dtype=int)
    node.current_allocs.copy_(parent_node.current_allocs)
    node.allocation_step = parent_node.allocation_step+1
    for qubit in alloc_qubits:
      node.current_allocs[qubit] = action # action means qubit is allocated to a core, so action = core
    # Node has reached end of allocation process
    if node.allocation_step == len(self.circuit.alloc_steps):
      node.terminal = True
    else:
      node.current_slice = self.circuit.alloc_steps[node.allocation_step][0]
      # Node corresponds to a time slice different than its parent's, set prev allocs to current allocs
      if parent_node.current_slice != node.current_slice:
        node.prev_allocs = node.current_allocs
        node.current_allocs = torch.empty_like(parent_node.current_allocs, dtype=int)
        node.core_caps = self.hardware.core_capacities
      else:
        node.prev_allocs = parent_node.prev_allocs
        node.core_caps = torch.empty_like(self.hardware.core_capacities)
        node.core_caps.copy_(parent_node.core_caps)
        node.core_caps[action] -= len(alloc_qubits)
        assert node.core_caps[action] >= 0
    self._expandNode(node)


  def _expandNode(self, node: Node):
    if node.terminal:
      return
    node.children = {}
    slice_idx_children, qubits_to_alloc = self.circuit.alloc_steps[node.allocation_step]
    for action in range(self.hardware.n_cores):
      if node.policy[action] == 0:
        continue
      child = MCTS.Node()
      node.core_caps[action] -= len(qubits_to_alloc)
      assert node.core_caps[action] >= 0, 'Not enough space in core to expand'
      child.policy, child.value_sum = self.__getNewPolicyAndValue(slice_idx=slice_idx_children,
                                                                  alloc_step=node.allocation_step+1,
                                                                  core_caps=node.core_caps)
      node.core_caps[action] += len(qubits_to_alloc)
      child.cost = self.__computeActionCost(node, action)
      node.children[action] = child
  

  def __computeActionCost(self, node: Node, action: int) -> int:
    if node.current_slice == 0:
      return 0
    _, qubits_to_alloc = self.circuit.alloc_steps[node.allocation_step]
    cost_sum = 0
    for q in qubits_to_alloc:
      cost_sum += self.hardware.core_connectivity[action, node.prev_allocs[q]]
    return cost_sum


  def __backprop(self, search_path: List[Node], node_value: float):
    # Reverse list order and pair items. For example [0,1,2,3] -> ((2,3), (1,2), (0,1))
    for node, next_node in zip(search_path[-2::-1], search_path[:0:-1]):
      node.value_sum += next_node.value + next_node.cost
      node.visit_count += 1
    if search_path[-1].terminal:
      search_path[-1].visit_count += 1
  
  
  def __UCB(self, node: Node, action: int) -> float:
    ''' Upper Confidence Bound with minus sign in CB to account for minimization of cost.

    For a nicely formatted version of this formula refer to Appendix B in Ref. [1]
    '''
    return (node.children[action].value + node.children[action].cost) - \
           node.policy[action]*sqrt(node.visit_count)/(1+node.children[action].visit_count) * \
              (self.cfg.ucb_c1 + log((node.visit_count + self.cfg.ucb_c2 + 1)/self.cfg.ucb_c2))

  
  def __selectChild(self, current_node: Node) -> Tuple[int, Node]:
    (_, action) = min((self.__UCB(current_node, a), a) for a in current_node.children.keys())
    return action, current_node.children[action]