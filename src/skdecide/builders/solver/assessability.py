# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from skdecide.core import D, autocastable

__all__ = ["Utilities", "QValues"]


class Utilities:
    """A solver must inherit this class if it can provide the utility function (i.e. value function)."""

    @autocastable
    def get_utility(self, observation: D.T_agent[D.T_observation]) -> D.T_value:
        """Get the estimated on-policy utility of the given observation.

        In mathematical terms, for a fully observable domain, this function estimates:
        $$V^\\pi(s)=\\underset{\\tau\\sim\\pi}{\\mathbb{E}}[R(\\tau)|s_0=s]$$
        where $\\pi$ is the current policy, any $\\tau=(s_0,a_0, s_1, a_1, ...)$ represents a trajectory sampled from
        the policy, $R(\\tau)$ is the return (cumulative reward) and $s_0$ the initial state for the trajectories.

        # Parameters
        observation: The observation to consider.

        # Returns
        The estimated on-policy utility of the given observation.
        """
        return self._get_utility(observation)

    def _get_utility(self, observation: D.T_agent[D.T_observation]) -> D.T_value:
        """Get the estimated on-policy utility of the given observation.

        In mathematical terms, for a fully observable domain, this function estimates:
        $$V^\\pi(s)=\\underset{\\tau\\sim\\pi}{\\mathbb{E}}[R(\\tau)|s_0=s]$$
        where $\\pi$ is the current policy, any $\\tau=(s_0,a_0, s_1, a_1, ...)$ represents a trajectory sampled from
        the policy, $R(\\tau)$ is the return (cumulative reward) and $s_0$ the initial state for the trajectories.

        # Parameters
        observation: The observation to consider.

        # Returns
        The estimated on-policy utility of the given observation.
        """
        raise NotImplementedError


class QValues(Utilities):
    """A solver must inherit this class if it can provide the Q function (i.e. action-value function)."""

    @autocastable
    def get_q_value(
        self,
        observation: D.T_agent[D.T_observation],
        action: D.T_agent[D.T_concurrency[D.T_event]],
    ) -> D.T_value:
        """Get the estimated on-policy Q value of the given observation and action.

        In mathematical terms, for a fully observable domain, this function estimates:
        $$Q^\\pi(s,a)=\\underset{\\tau\\sim\\pi}{\\mathbb{E}}[R(\\tau)|s_0=s,a_0=a]$$
        where $\\pi$ is the current policy, any $\\tau=(s_0,a_0, s_1, a_1, ...)$ represents a trajectory sampled from
        the policy, $R(\\tau)$ is the return (cumulative reward) and $s_0$/$a_0$ the initial state/action for the
        trajectories.

        # Parameters
        observation: The observation to consider.
        action: The action to consider.

        # Returns
        The estimated on-policy Q value of the given observation and action.
        """
        return self._get_q_value(observation, action)

    def _get_q_value(
        self,
        observation: D.T_agent[D.T_observation],
        action: D.T_agent[D.T_concurrency[D.T_event]],
    ) -> D.T_value:
        """Get the estimated on-policy Q value of the given observation and action.

        In mathematical terms, for a fully observable domain, this function estimates:
        $$Q^\\pi(s,a)=\\underset{\\tau\\sim\\pi}{\\mathbb{E}}[R(\\tau)|s_0=s,a_0=a]$$
        where $\\pi$ is the current policy, any $\\tau=(s_0,a_0, s_1, a_1, ...)$ represents a trajectory sampled from
        the policy, $R(\\tau)$ is the return (cumulative reward) and $s_0$/$a_0$ the initial state/action for the
        trajectories.

        # Parameters
        observation: The observation to consider.
        action: The action to consider.

        # Returns
        The estimated on-policy Q value of the given observation and action.
        """
        raise NotImplementedError
