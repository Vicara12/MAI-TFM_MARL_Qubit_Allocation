import abc
from typing import Callable, List, Union

import torch
import torch.nn.functional as F

from src.utils.other_utils import gather_by_index


class AgentHandler(abc.ABC):
    """Base class for agent handlers. Handles conflicts between agents.
    By default, one occurrence is always kept (i.e. one agent among agents
    that selected the same action is selected).

    Args:
        mask_all: If True, the all occurrences of the same value will be masked, i.e. no agent will select it.
        exclude_values: If provided, the values in the actions that are in this list will not be masked.
        return_none_mask: If True, the mask will be None. This may be useful for loss computation.
    """

    def __init__(
        self,
        mask_all: bool = False,
        exclude_values: Union[torch.Tensor, int, List] = None,
        return_none_mask: bool = False,
    ):
        super(AgentHandler, self).__init__()
        self.mask_all = mask_all
        self.exclude_values = exclude_values
        self.return_none_mask = return_none_mask

    @abc.abstractmethod
    def _preprocess_actions(
        self, actions: torch.Tensor, td, probs: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Preprocesses actions such that first action in order to appear with an index will be selected"""
        raise NotImplementedError("Subclasses must implement this method.")

    def __call__(
        self,
        actions: torch.Tensor,
        probs: torch.Tensor,
        core_capacities: torch.Tensor,
        agent_demands: torch.Tensor | None = None,
        replacement_value: Union[torch.Tensor, int, None] = None,
        buffer_index: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Greedy capacity-aware conflict handler for the QA setting (no batch).

        Agents are processed in descending probability of their chosen action. If an
        agent would exceed the remaining capacity of a core, it is masked and its
        action replaced (default: buffer core).
        """
        if probs is None:
            raise ValueError("probs must be provided for conflict handling")

        device = actions.device
        if actions.ndim != 1:
            raise ValueError(f"AgentHandlerQA expects 1-D actions, got shape {tuple(actions.shape)}")
        buffer_idx = buffer_index if buffer_index is not None else core_capacities.numel() - 1
        replacement = buffer_idx if replacement_value is None else replacement_value

        demands = agent_demands
        if demands.numel() != actions.numel():
            raise ValueError(
                f"Agent demand length {demands.numel()} does not match actions length {actions.numel()}"
            )

        sorted_actions, order_indices = self._preprocess_actions(actions, td=None, probs=probs, **kwargs)
        demands_sorted = demands.gather(0, order_indices)

        # Check demand vs capacity
        num_cores = core_capacities.numel()
        capacities = core_capacities.to(device)
        capacities_with_buffer = capacities.clone()
        if capacities_with_buffer.is_floating_point():
            buffer_cap = torch.finfo(capacities_with_buffer.dtype).max
        else:
            buffer_cap = torch.iinfo(capacities_with_buffer.dtype).max
        capacities_with_buffer[buffer_idx] = buffer_cap

        action_one_hot = F.one_hot(sorted_actions, num_classes=num_cores).to(demands_sorted.dtype)
        demand_matrix = demands_sorted.unsqueeze(-1) * action_one_hot
        cumulative_demand = demand_matrix.cumsum(dim=0)

        cumulative_for_action = cumulative_demand.gather(1, sorted_actions.unsqueeze(-1)).squeeze(-1)
        capacity_for_action = capacities_with_buffer.gather(0, sorted_actions)

        over_capacity = cumulative_for_action > capacity_for_action
        mask_sorted = over_capacity & (sorted_actions != buffer_idx)

        inverse_order = order_indices.argsort(dim=0)
        mask = mask_sorted.gather(0, inverse_order)

        actions = actions.clone()
        if isinstance(replacement, int):
            actions[mask] = replacement
        else:
            actions[mask] = replacement[mask]

        halting_ratio = mask.float().mean() if mask.numel() > 0 else torch.tensor(0.0, device=device)
        return actions, None if self.return_none_mask else mask, halting_ratio


class HighestProbabilityAgentHandler(AgentHandler):
    """Highest probability agent in dim 1 is selected in case of conflicts"""

    def _preprocess_actions(
        self, actions: torch.Tensor, td, probs, randomize: bool = False
    ) -> torch.Tensor:
        # sort indices by probability
        action_probs = gather_by_index(probs, actions, dim=-1)
        if randomize:
            # Add Gumbel noise to probabilities to randomize selection
            # This may be useful for exploration
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(action_probs)))
            action_probs += gumbel_noise
        # Sort indices by action probabilities
        _, indices = torch.sort(action_probs, dim=0, descending=True, stable=True)
        sorted_actions = gather_by_index(actions, indices, dim=0)
        return sorted_actions, indices



class NoHandler(AgentHandler):
    """No handler is used, i.e. all agents can select the same action. The mask is always None."""

    def __call__(self, actions: torch.Tensor, *args, **kwargs):
        return actions, None, torch.tensor(0.0, device=actions.device)


AGENT_HANDLER_REGISTRY = {
    "highprob": HighestProbabilityAgentHandler,
    "none": NoHandler,
}


def get_agent_handler(
    name: str, registry: dict = AGENT_HANDLER_REGISTRY, **config
) -> Callable:
    return registry[name](**config)
