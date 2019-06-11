from typing import Generic

from airlaps.core import T_observation, T_event, Memory

__all__ = ['UtilitySolver', 'QSolver']


class UtilitySolver(Generic[T_observation, T_event]):
    """A solver must inherit this class if it can provide the utility function (i.e. value function)."""

    def get_utility(self, memory: Memory[T_observation]) -> float:
        """Get the on-policy utility of the given memory.

        In mathematical terms, assuming a Markovian domain (memory only holds last state), this function represents:
        $$
        V^\\pi(s)=\\underset{\\tau\\sim\\pi}{\\mathbb{E}}[R(\\tau)|s_0=s]
        $$
        where $\\pi$ is the current policy, any $\\tau=(s_0,a_0, s_1, a_1, ...)$ represents a trajectory sampled from
        the policy, $R(\\tau)$ is the return (cumulative reward) and $s_0$ the initial state for the trajectories.

        # Parameters
        memory: The memory to consider.

        # Returns
        The on-policy utility of the given memory.
        """
        raise NotImplementedError


class QSolver(UtilitySolver[T_observation, T_event]):
    """A solver must inherit this class if it can provide the Q function (i.e. action-value function)."""

    def get_q_value(self, memory: Memory[T_observation], event: T_event) -> float:
        """Get the on-policy Q value of the given memory and event.

        In mathematical terms, assuming a Markovian domain (memory only holds last state) with only actions (no pure
        events), this function represents:
        $$
        Q^\\pi(s,a)=\\underset{\\tau\\sim\\pi}{\\mathbb{E}}[R(\\tau)|s_0=s,a_0=a]
        $$
        where $\\pi$ is the current policy, any $\\tau=(s_0,a_0, s_1, a_1, ...)$ represents a trajectory sampled from
        the policy, $R(\\tau)$ is the return (cumulative reward) and $s_0$/$a_0$ the initial state/action for the
        trajectories.

        # Parameters
        memory: The memory to consider.
        event: The event to consider.

        # Returns
        The on-policy Q value of the given memory and event.
        """
        raise NotImplementedError
