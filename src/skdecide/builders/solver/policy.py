# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from skdecide.core import D, Distribution, SingleValueDistribution, autocastable

__all__ = ["Policies", "UncertainPolicies", "DeterministicPolicies"]


class Policies:
    """A solver must inherit this class if it computes a stochastic policy as part of the solving process."""

    @autocastable
    def sample_action(
        self, observation: D.T_agent[D.T_observation]
    ) -> D.T_agent[D.T_concurrency[D.T_event]]:
        """Sample an action for the given observation (from the solver's current policy).

        # Parameters
        observation: The observation for which an action must be sampled.

        # Returns
        The sampled action.
        """
        return self._sample_action(observation)

    def _sample_action(
        self, observation: D.T_agent[D.T_observation]
    ) -> D.T_agent[D.T_concurrency[D.T_event]]:
        """Sample an action for the given observation (from the solver's current policy).

        # Parameters
        observation: The observation for which an action must be sampled.

        # Returns
        The sampled action.
        """
        raise NotImplementedError

    @autocastable
    def is_policy_defined_for(self, observation: D.T_agent[D.T_observation]) -> bool:
        """Check whether the solver's current policy is defined for the given observation.

        # Parameters
        observation: The observation to consider.

        # Returns
        True if the policy is defined for the given observation memory (False otherwise).
        """
        return self._is_policy_defined_for(observation)

    def _is_policy_defined_for(self, observation: D.T_agent[D.T_observation]) -> bool:
        """Check whether the solver's current policy is defined for the given observation.

        # Parameters
        observation: The observation to consider.

        # Returns
        True if the policy is defined for the given observation memory (False otherwise).
        """
        raise NotImplementedError


class UncertainPolicies(Policies):
    """A solver must inherit this class if it computes a stochastic policy (providing next action distribution
    explicitly) as part of the solving process."""

    def _sample_action(
        self, observation: D.T_agent[D.T_observation]
    ) -> D.T_agent[D.T_concurrency[D.T_event]]:
        return self._get_next_action_distribution(observation).sample()

    @autocastable
    def get_next_action_distribution(
        self, observation: D.T_agent[D.T_observation]
    ) -> Distribution[D.T_agent[D.T_concurrency[D.T_event]]]:
        """Get the probabilistic distribution of next action for the given observation (from the solver's current
        policy).

        # Parameters
        observation: The observation to consider.

        # Returns
        The probabilistic distribution of next action.
        """
        return self._get_next_action_distribution(observation)

    def _get_next_action_distribution(
        self, observation: D.T_agent[D.T_observation]
    ) -> Distribution[D.T_agent[D.T_concurrency[D.T_event]]]:
        """Get the probabilistic distribution of next action for the given observation (from the solver's current
        policy).

        # Parameters
        observation: The observation to consider.

        # Returns
        The probabilistic distribution of next action.
        """
        raise NotImplementedError


class DeterministicPolicies(UncertainPolicies):
    """A solver must inherit this class if it computes a deterministic policy as part of the solving process."""

    def _get_next_action_distribution(
        self, observation: D.T_agent[D.T_observation]
    ) -> Distribution[D.T_agent[D.T_concurrency[D.T_event]]]:
        return SingleValueDistribution(self._get_next_action(observation))

    @autocastable
    def get_next_action(
        self, observation: D.T_agent[D.T_observation]
    ) -> D.T_agent[D.T_concurrency[D.T_event]]:
        """Get the next deterministic action (from the solver's current policy).

        # Parameters
        observation: The observation for which next action is requested.

        # Returns
        The next deterministic action.
        """
        return self._get_next_action(observation)

    def _get_next_action(
        self, observation: D.T_agent[D.T_observation]
    ) -> D.T_agent[D.T_concurrency[D.T_event]]:
        """Get the next deterministic action (from the solver's current policy).

        # Parameters
        observation: The observation for which next action is requested.

        # Returns
        The next deterministic action.
        """
        raise NotImplementedError
