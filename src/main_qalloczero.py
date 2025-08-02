from utils.timer import Timer
from qalloczero.alg.replaybuff import ReplayBuffer
import torch


def add(n_elms, rb):
  states  = torch.rand((n_elms,3))
  actions = torch.rand((n_elms,1))
  rewards = torch.rand((n_elms,1))
  print(f"{states =}\n{actions =}\n{rewards}")
  rb.push(states=states, actions=actions, rewards=rewards)


def main():
  r = ReplayBuffer(10, 3)
  
  r.setSamplingMode()
  stats, acts, rwds = r.sample(2, "cpu")
  print(f"\n{stats =}\n{acts =}\n{rwds =}")


if __name__ == "__main__":
  main()