from typing import List, Self, Tuple, Dict, Union
from math import sqrt, ln
from copy import copy
from dataclasses import dataclass
import torch
from qalloczero.models.inferenceserver import InferenceServer



class MCTS:
  ''' A class that performs a round of Monte Carlo Tree Search driven by DL heuristics.

  Args:
    - init_repr: initial representation of the state (s_0 in Ref. [1]).

  References:
    [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model]
    (https://arxiv.org/abs/1911.08265)
      Julian Schrittwieser et. al. 2020.
  '''

  @dataclass
  class Node:
    core_caps: List[int]
    state: torch.Tensor
    policy: List[float]
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


  def __init__(self, init_repr: torch.Tensor, core_capacities: torch.Tensor, config: Config):
    self.core_capacities = core_capacities
    self.init_repr = init_repr
    self.n_cores = len(core_capacities)
    self.config = copy(config)
    self.ucb_c1 = 1.25  # As in Ref. [1]
    self.ucb_c2 = 19652 # As in Ref. [1]
    self.root = self.__buildRoot(init_repr=init_repr, core_capacities=core_capacities)
  

  def run(self, num_sims: int):
    for _ in range(num_sims):
      node = self.root
      search_path = [node]
      last_action = None
      while node.expanded:
        last_action, node = self.__selectChild(current_node=node)
        search_path.append(node)
        node = node.children[last_action]
      father = search_path[-2]
      node_value = self.__expandNode(node=node, father_node=father, action=last_action)
      self.__backprop(node_value)
    return self.__selectAction(self.root)
  

  @staticmethod
  def __selectAction(node: Node) -> int:
    visit_counts = torch.tensor((child.visit_count for child in node.children.values()),
                                dtype=torch.float)
    # Return a random core allocation with probability proportional to the visit count of each of
    # the node's children
    return torch.multinomial(visit_counts, num_samples=1).item()


  def __getNewPolicyAndValue(self,
                             state: torch.Tensor,
                             core_caps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    pol, v = InferenceServer.PRED_MODEL(state, core_caps)
    # Add exploration noise to the priors
    dir_noise = torch.distributions.Dirichlet(self.config.dirichlet_alpha * torch.ones_like(pol)).sample()
    pol = (1 - self.config.noise)*pol + self.config.noise*dir_noise
    pol[core_caps == 0] = 0 # Set prior of exploring a core without capacity to zero
    return pol, v
  

  def __getNewStateAndReward(self, state: torch.Tensor, action: int) -> Tuple[torch.Tensor, float]:
    new_state, r = InferenceServer.DYN_MODEL(state, action)
    return new_state, r


  def __buildRoot(self, init_repr: torch.Tensor, core_caps: torch.Tensor) -> Node:
    pol, v = self.__getNewPolicyAndValue(state=init_repr, core_caps=core_caps)
    return MCTS.Node(
      core_caps=core_caps.copy_(),
      state=init_repr,
      policy=pol,
      children={c: MCTS.Node() for c in range(self.n_cores) if core_caps[c] != 0}
    )


  def __expandNode(self,
                   node: Node,
                   father_node: Union[Node, None],
                   action: Union[int, None]) -> float:
    node.core_caps = father_node.core_caps.copy_()
    node.core_caps[action] -= 1
    assert node.core_caps[action] >= 0
    node.state, node.reward = self.__getNewStateAndReward(state=father_node.state, action=action)
    node.policy, value  = self.__getNewPolicyAndValue(node.state, node.core_caps)
    node.children={c: MCTS.Node() for c in range(self.n_cores) if node.core_caps[c] != 0}
    return value


  def __backprop(self, search_path: List[Node], node_value: float):
    for node in search_path:
      node.value_sum += node_value
      node.visit_count += 1
      node_value = node.reward + self.config.discount_factor*node_value
  
  
  def __UCB(self, node: Node, action: int) -> int:
    ''' Upper Confidence Bound.

    For a nicely formatted version of this formula refer to Appendix B in Ref. [1]
    '''
    return node[action].value + \
           node.policy[action]*sqrt(node.visit_count)/(1+node[action].visit_count) * \
              (self.ucb_c1 + ln((node.visit_count + self.ucb_c2 + 1)/self.ucb_c2))

  
  def __selectChild(self, current_node: Node) -> Tuple[int, Node]:
    _, action, child = max((self.__UCB(n, a) for a, n in current_node.children.items()))
    return action, child