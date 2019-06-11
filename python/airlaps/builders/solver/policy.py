from typing import Generic

from airlaps.core import T_observation, T_event, Distribution, SingleValueDistribution, Memory

__all__ = ['PolicySolver', 'UncertainPolicySolver', 'DeterministicPolicySolver']


class PolicySolver(Generic[T_observation, T_event]):
    """A solver must inherit this class if it computes a stochastic policy as part of the solving process."""

    def sample_action(self, memory: Memory[T_observation]) -> T_event:
        """Sample an action for the given observation memory (from the solver's current policy).

        # Parameters
        memory: The observation memory for which an action must be sampled.

        # Returns
        The sampled action.
        """
        raise NotImplementedError

    def is_policy_defined_for(self, memory: Memory[T_observation]) -> bool:
        """Check whether the solver's current policy is defined for the given observation memory.

        # Parameters
        memory: The observation memory to consider.

        # Returns
        True if the policy is defined for the given observation memory (False otherwise).
        """
        raise NotImplementedError


class UncertainPolicySolver(PolicySolver[T_observation, T_event]):
    """A solver must inherit this class if it computes a stochastic policy (providing next action distribution
    explicitly) as part of the solving process."""

    def sample_action(self, memory: Memory[T_observation]) -> T_event:
        return self.get_next_action_distribution(memory).sample()

    def get_next_action_distribution(self, memory: Memory[T_observation]) -> Distribution[T_event]:
        """Get the probabilistic distribution of next action for the given observation memory (from the solver's current
        policy).
        
        # Parameters
        memory: The observation memory to consider.
        
        # Returns
        The probabilistic distribution of next action.
        """
        raise NotImplementedError


class DeterministicPolicySolver(UncertainPolicySolver[T_observation, T_event]):
    """A solver must inherit this class if it computes a deterministic policy as part of the solving process."""

    def get_next_action_distribution(self, memory: Memory[T_observation]) -> Distribution[T_event]:
        return SingleValueDistribution(self.get_next_action(memory))

    def get_next_action(self, memory: Memory[T_observation]) -> T_event:
        """Get the next deterministic action (from the solver's current policy).

        # Parameters
        memory: The observation memory for which next action is requested.

        # Returns
        The next deterministic action.
        """
        raise NotImplementedError
