from typing import List, Self, Tuple, Dict, Union
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
    current_allocs: Dict[int, int]
    prev_allocs: Dict[int, int] # Allocations in the previous time slice
    core_caps: List[int]
    # State attributes
    allocation_step: int
    terminal: bool = False
    current_slice: int
    # RL attributes
    policy: torch.Tensor
    value_sum: float = 0
    visit_count: int = 0
    reward:      int = 0
    children: Dict[int,Self] = None

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



  def __init__(self, circuit_embs: torch.Tensor, circuit: Circuit, hardware: Hardware, config: Config):
    self.circuit_embs = circuit_embs
    self.circuit = circuit
    self.hardware = hardware
    self.cfg = config
    self.root = self.__buildRoot()


  def run(self, target_tree_size: int):
    # Visit count is equal to the current size of the tree
    num_sims = target_tree_size - self.root.visit_count
    for _ in range(num_sims):
      node = self.root
      search_path = [node]
      last_action = None
      while node.expanded and not node.terminal:
        last_action, node = self.__selectChild(current_node=node)
        search_path.append(node)
        node = node.children[last_action]
      if not node.terminal:
        node_value = self.__expandNode(node=node, parent_node=search_path[-2], action=last_action)
      else:
        node_value = node.value
      self.__backprop(node_value)
    action = self.__selectAction(self.root)
    self.root = self.root.children[action]
    return action


  @staticmethod
  def __selectAction(node: Node) -> int:
    visit_counts = torch.tensor((child.visit_count for child in node.children.values()),
                                dtype=torch.float)
    # Return a random core allocation with probability proportional to the visit count of each of
    # the node's children
    return torch.multinomial(visit_counts, num_samples=1).item()


  def __getNewPolicyAndValue(self, node: Node) -> Tuple[torch.Tensor, torch.Tensor]:
    state = self.circuit_embs[node.current_slice]
    pol, v = InferenceServer.inference(state, node.core_caps)
    # Add exploration noise to the priors
    dir_noise = torch.distributions.Dirichlet(self.cfg.dirichlet_alpha * torch.ones_like(pol)).sample()
    pol = (1 - self.cfg.noise)*pol + self.cfg.noise*dir_noise
    # Set prior of cores that do not have space for this alloc to zero
    n_qubits = len(self.circuit.alloc_steps[node.allocation_step][1])
    pol[node.core_caps < n_qubits] = 0
    return pol, v


  def __buildRoot(self) -> Node:
    pol, v = self.__getNewPolicyAndValue(state=self.circuit_embs[0],
                                         core_caps=self.hardware.core_capacities)
    return MCTS.Node(
      current_allocs={},
      prev_allocs={},
      core_caps=self.hardware.core_capacities,
      allocation_step=0,
      allocation_step=0,
      policy=pol,
      children={c: MCTS.Node() for c in range(self.hardware.n_cores)}
    )


  def __expandNode(self,
                   node: Node,
                   parent_node: Union[Node, None],
                   action: Union[int, None]) -> float:
    (slice_parent, alloc_qubits) = self.circuit.alloc_steps[parent_node.allocation_step]
    node.current_allocs = copy(parent_node.current_allocs)
    node.allocation_step = parent_node.allocation_step+1
    for qubit in alloc_qubits:
      node.current_allocs[qubit] = action # action means qubit is allocated to a core, so action = core
    if slice_parent != 0:
      for qubit in alloc_qubits:
        node.reward += self.hardware.core_connectivity[action, parent_node.prev_allocs[qubit]]
    # Node has reached end of allocation process
    if node.allocation_step == len(self.circuit.alloc_steps):
      node.terminal = True
    else:
      node.current_slice = self.circuit.alloc_steps[node.allocation_step][0]
      # Node corresponds to a time slice different than its parent's, set prev allocs to current allocs
      if parent_node.current_slice != node.current_slice:
        node.prev_allocs = node.current_allocs
        node.current_allocs = {}
        node.core_caps = self.hardware.core_capacities
      else:
        node.prev_allocs = parent_node.prev_allocs
        node.core_caps = copy(parent_node.core_caps)
        node.core_caps[action] -= len(alloc_qubits)
        assert node.core_caps[action] >= 0
    node.policy, value  = self.__getNewPolicyAndValue(node)
    node.children={c: MCTS.Node() for c in range(self.n_cores) if node.core_caps[c] != 0}
    return value


  def __backprop(self, search_path: List[Node], node_value: float):
    for node in search_path:
      node.value_sum += node_value
      node.visit_count += 1
      node_value = node.reward + self.cfg.discount_factor*node_value
  
  
  def __UCB(self, node: Node, action: int) -> int:
    ''' Upper Confidence Bound.

    For a nicely formatted version of this formula refer to Appendix B in Ref. [1]
    '''
    return node[action].value + \
           node.policy[action]*sqrt(node.visit_count)/(1+node[action].visit_count) * \
              (self.cfg.ucb_c1 + log((node.visit_count + self.cfg.ucb_c2 + 1)/self.cfg.ucb_c2))

  
  def __selectChild(self, current_node: Node) -> Tuple[int, Node]:
    _, action, child = max((self.__UCB(n, a) for a, n in current_node.children.items()))
    return action, child