# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import functools
from typing import Optional

from skdecide.core import (
    D,
    DiscreteDistribution,
    Distribution,
    EnvironmentOutcome,
    SingleValueDistribution,
    TransitionOutcome,
    Value,
    autocastable,
)

__all__ = [
    "Environment",
    "Simulation",
    "UncertainTransitions",
    "EnumerableTransitions",
    "DeterministicTransitions",
]


class Environment:
    """A domain must inherit this class if agents interact with it like a black-box environment.

    Black-box environment examples include: the real world, compiled ATARI games, etc.

    !!! tip
        Environment domains are typically stateful: they must keep the current state or history in their memory to
        compute next steps (automatically done by default in the #_memory attribute).
    """

    @autocastable
    def step(
        self, action: D.T_agent[D.T_concurrency[D.T_event]]
    ) -> EnvironmentOutcome[
        D.T_agent[D.T_observation],
        D.T_agent[Value[D.T_value]],
        D.T_agent[D.T_predicate],
        D.T_agent[D.T_info],
    ]:
        """Run one step of the environment's dynamics.

        By default, #Environment.step() provides some boilerplate code and internally calls #Environment._step() (which
        returns a transition outcome). The boilerplate code automatically stores next state into the #_memory attribute
        and samples a corresponding observation.

        !!! tip
            Whenever an existing environment needs to be wrapped instead of implemented fully in scikit-decide (e.g. compiled
            ATARI games), it is recommended to overwrite #Environment.step() to call the external environment and not
            use the #Environment._step() helper function.

        !!! warning
            Before calling #Environment.step() the first time or when the end of an episode is
            reached, #Initializable.reset() must be called to reset the environment's state.

        # Parameters
        action: The action taken in the current memory (state or history) triggering the transition.

        # Returns
        The environment outcome of this step.
        """
        return self._step(action)

    def _step(
        self, action: D.T_agent[D.T_concurrency[D.T_event]]
    ) -> EnvironmentOutcome[
        D.T_agent[D.T_observation],
        D.T_agent[Value[D.T_value]],
        D.T_agent[D.T_predicate],
        D.T_agent[D.T_info],
    ]:
        """Run one step of the environment's dynamics.

        By default, #Environment._step() provides some boilerplate code and internally
        calls #Environment._state_step() (which returns a transition outcome). The boilerplate code automatically stores
        next state into the #_memory attribute and samples a corresponding observation.

        !!! tip
            Whenever an existing environment needs to be wrapped instead of implemented fully in scikit-decide (e.g. compiled
            ATARI games), it is recommended to overwrite #Environment._step() to call the external environment and not
            use the #Environment._state_step() helper function.

        !!! warning
            Before calling #Environment._step() the first time or when the end of an episode is
            reached, #Initializable._reset() must be called to reset the environment's state.

        # Parameters
        action: The action taken in the current memory (state or history) triggering the transition.

        # Returns
        The environment outcome of this step.
        """
        transition_outcome = self._state_step(action)
        next_state = transition_outcome.state
        observation = self._get_observation_distribution(next_state, action).sample()
        if self._get_memory_maxlen() == 1:
            self._memory = next_state
        elif self._get_memory_maxlen() > 1:
            self._memory.append(next_state)
        return EnvironmentOutcome(
            observation,
            transition_outcome.value,
            transition_outcome.termination,
            transition_outcome.info,
        )

    def _state_step(
        self, action: D.T_agent[D.T_concurrency[D.T_event]]
    ) -> TransitionOutcome[
        D.T_state,
        D.T_agent[Value[D.T_value]],
        D.T_agent[D.T_predicate],
        D.T_agent[D.T_info],
    ]:
        """Compute one step of the transition's dynamics.

        This is a helper function called by default from #Environment._step(). It focuses on the state level, as opposed
        to the observation one for the latter.

        # Parameters
        action: The action taken in the current memory (state or history) triggering the transition.

        # Returns
        The transition outcome of this step.
        """
        raise NotImplementedError


class Simulation(Environment):
    """A domain must inherit this class if agents interact with it like a simulation.

    Compared to pure environment domains, simulation ones have the additional ability to sample transitions from any
    given state.

    !!! tip
        Simulation domains are typically stateless: they do not need to store the current state or history in memory
        since it is usually passed as parameter of their functions. By default, they only become stateful whenever they
        are used as environments (e.g. via #Initializable.reset() and #Environment.step() functions).
    """

    def _state_step(
        self, action: D.T_agent[D.T_concurrency[D.T_event]]
    ) -> TransitionOutcome[
        D.T_state,
        D.T_agent[Value[D.T_value]],
        D.T_agent[D.T_predicate],
        D.T_agent[D.T_info],
    ]:
        return self._state_sample(self._memory, action)

    @autocastable
    def set_memory(self, memory: D.T_memory[D.T_state]) -> None:
        """Set internal memory attribute #_memory to given one.

        This can be useful to set a specific "starting point" before doing a rollout with successive #Environment.step()
        calls.

        # Parameters
        memory: The memory to set internally.

        # Example
        ```python
        # Set simulation_domain memory to my_state (assuming Markovian domain)
        simulation_domain.set_memory(my_state)

        # Start a 100-steps rollout from here (applying my_action at every step)
        for _ in range(100):
            simulation_domain.step(my_action)
        ```
        """
        return self._set_memory(memory)

    def _set_memory(self, memory: D.T_memory[D.T_state]) -> None:
        """Set internal memory attribute #_memory to given one.

        This can be useful to set a specific "starting point" before doing a rollout with
        successive #Environment._step() calls.

        # Parameters
        memory: The memory to set internally.

        # Example
        ```python
        # Set simulation_domain memory to my_state (assuming Markovian domain)
        simulation_domain._set_memory(my_state)

        # Start a 100-steps rollout from here (applying my_action at every step)
        for _ in range(100):
            simulation_domain._step(my_action)
        ```
        """
        self._memory = memory

    @autocastable
    def sample(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
    ) -> EnvironmentOutcome[
        D.T_agent[D.T_observation],
        D.T_agent[Value[D.T_value]],
        D.T_agent[D.T_predicate],
        D.T_agent[D.T_info],
    ]:
        """Sample one transition of the simulator's dynamics.

        By default, #Simulation.sample() provides some boilerplate code and internally calls #Simulation._sample()
        (which returns a transition outcome). The boilerplate code automatically samples an observation corresponding to
        the sampled next state.

        !!! tip
            Whenever an existing simulator needs to be wrapped instead of implemented fully in scikit-decide (e.g. a
            simulator), it is recommended to overwrite #Simulation.sample() to call the external simulator and not use
            the #Simulation._sample() helper function.

        # Parameters
        memory: The source memory (state or history) of the transition.
        action: The action taken in the given memory (state or history) triggering the transition.

        # Returns
        The environment outcome of the sampled transition.
        """
        return self._sample(memory, action)

    def _sample(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
    ) -> EnvironmentOutcome[
        D.T_agent[D.T_observation],
        D.T_agent[Value[D.T_value]],
        D.T_agent[D.T_predicate],
        D.T_agent[D.T_info],
    ]:
        """Sample one transition of the simulator's dynamics.

        By default, #Simulation._sample() provides some boilerplate code and internally
        calls #Simulation._state_sample() (which returns a transition outcome). The boilerplate code automatically
        samples an observation corresponding to the sampled next state.

        !!! tip
            Whenever an existing simulator needs to be wrapped instead of implemented fully in scikit-decide (e.g. a
            simulator), it is recommended to overwrite #Simulation._sample() to call the external simulator and not use
            the #Simulation._state_sample() helper function.

        # Parameters
        memory: The source memory (state or history) of the transition.
        action: The action taken in the given memory (state or history) triggering the transition.

        # Returns
        The environment outcome of the sampled transition.
        """
        transition_outcome = self._state_sample(memory, action)
        next_state = transition_outcome.state
        observation = self._get_observation_distribution(next_state, action).sample()
        return EnvironmentOutcome(
            observation,
            transition_outcome.value,
            transition_outcome.termination,
            transition_outcome.info,
        )

    def _state_sample(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
    ) -> TransitionOutcome[
        D.T_state,
        D.T_agent[Value[D.T_value]],
        D.T_agent[D.T_predicate],
        D.T_agent[D.T_info],
    ]:
        """Compute one sample of the transition's dynamics.

        This is a helper function called by default from #Simulation._sample(). It focuses on the state level, as
        opposed to the observation one for the latter.

        # Parameters
        memory: The source memory (state or history) of the transition.
        action: The action taken in the given memory (state or history) triggering the transition.

        # Returns
        The transition outcome of the sampled transition.
        """
        raise NotImplementedError


class UncertainTransitions(Simulation):
    """A domain must inherit this class if its dynamics is uncertain and provided as a white-box model.

    Compared to pure simulation domains, uncertain transition ones provide in addition the full probability distribution
    of next states given a memory and action.

    !!! tip
        Uncertain transition domains are typically stateless: they do not need to store the current state or history in
        memory since it is usually passed as parameter of their functions. By default, they only become stateful
        whenever they are used as environments (e.g. via #Initializable.reset() and #Environment.step() functions).
    """

    def _state_sample(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
    ) -> TransitionOutcome[
        D.T_state,
        D.T_agent[Value[D.T_value]],
        D.T_agent[D.T_predicate],
        D.T_agent[D.T_info],
    ]:
        next_state = self._get_next_state_distribution(memory, action).sample()
        value = self._get_transition_value(memory, action, next_state)
        # Termination could be inferred using get_next_state_distribution based on next_state,
        # but would introduce multiple constraints on class definitions
        termination = self._is_terminal(next_state)
        return TransitionOutcome(next_state, value, termination, None)

    @autocastable
    def get_next_state_distribution(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
    ) -> Distribution[D.T_state]:
        """Get the probability distribution of next state given a memory and action.

        # Parameters
        memory: The source memory (state or history) of the transition.
        action: The action taken in the given memory (state or history) triggering the transition.

        # Returns
        The probability distribution of next state.
        """
        return self._get_next_state_distribution(memory, action)

    def _get_next_state_distribution(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
    ) -> Distribution[D.T_state]:
        """Get the probability distribution of next state given a memory and action.

        # Parameters
        memory: The source memory (state or history) of the transition.
        action: The action taken in the given memory (state or history) triggering the transition.

        # Returns
        The probability distribution of next state.
        """
        raise NotImplementedError

    @autocastable
    def get_transition_value(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
        next_state: Optional[D.T_state] = None,
    ) -> D.T_agent[Value[D.T_value]]:
        """Get the value (reward or cost) of a transition.

        The transition to consider is defined by the function parameters.

        !!! tip
            If this function never depends on the next_state parameter for its computation, it is recommended to
            indicate it by overriding #UncertainTransitions._is_transition_value_dependent_on_next_state_() to return
            False. This information can then be exploited by solvers to avoid computing next state to evaluate a
            transition value (more efficient).

        # Parameters
        memory: The source memory (state or history) of the transition.
        action: The action taken in the given memory (state or history) triggering the transition.
        next_state: The next state in which the transition ends (if needed for the computation).

        # Returns
        The transition value (reward or cost).
        """
        return self._get_transition_value(memory, action, next_state)

    def _get_transition_value(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
        next_state: Optional[D.T_state] = None,
    ) -> D.T_agent[Value[D.T_value]]:
        """Get the value (reward or cost) of a transition.

        The transition to consider is defined by the function parameters.

        !!! tip
            If this function never depends on the next_state parameter for its computation, it is recommended to
            indicate it by overriding #UncertainTransitions._is_transition_value_dependent_on_next_state_() to return
            False. This information can then be exploited by solvers to avoid computing next state to evaluate a
            transition value (more efficient).

        # Parameters
        memory: The source memory (state or history) of the transition.
        action: The action taken in the given memory (state or history) triggering the transition.
        next_state: The next state in which the transition ends (if needed for the computation).

        # Returns
        The transition value (reward or cost).
        """
        raise NotImplementedError

    @autocastable
    def is_transition_value_dependent_on_next_state(self) -> bool:
        """Indicate whether get_transition_value() requires the next_state parameter for its computation (cached).

        By default, #UncertainTransitions.is_transition_value_dependent_on_next_state() internally
        calls #UncertainTransitions._is_transition_value_dependent_on_next_state_() the first time and automatically
        caches its value to make future calls more efficient (since the returned value is assumed to be constant).

        # Returns
        True if the transition value computation depends on next_state (False otherwise).
        """
        return self._is_transition_value_dependent_on_next_state()

    @functools.lru_cache()
    def _is_transition_value_dependent_on_next_state(self) -> bool:
        """Indicate whether _get_transition_value() requires the next_state parameter for its computation (cached).

        By default, #UncertainTransitions._is_transition_value_dependent_on_next_state() internally
        calls #UncertainTransitions._is_transition_value_dependent_on_next_state_() the first time and automatically
        caches its value to make future calls more efficient (since the returned value is assumed to be constant).

        # Returns
        True if the transition value computation depends on next_state (False otherwise).
        """
        return self._is_transition_value_dependent_on_next_state_()

    def _is_transition_value_dependent_on_next_state_(self) -> bool:
        """Indicate whether _get_transition_value() requires the next_state parameter for its computation.

        This is a helper function called by default
        from #UncertainTransitions._is_transition_value_dependent_on_next_state(), the difference being that the result
        is not cached here.

        !!! tip
            The underscore at the end of this function's name is a convention to remind that its result should be
            constant.

        # Returns
        True if the transition value computation depends on next_state (False otherwise).
        """
        return True

    @autocastable
    def is_terminal(self, state: D.T_state) -> D.T_agent[D.T_predicate]:
        """Indicate whether a state is terminal.

        A terminal state is a state with no outgoing transition (except to itself with value 0).

        # Parameters
        state: The state to consider.

        # Returns
        True if the state is terminal (False otherwise).
        """
        return self._is_terminal(state)

    def _is_terminal(self, state: D.T_state) -> D.T_agent[D.T_predicate]:
        """Indicate whether a state is terminal.

        A terminal state is a state with no outgoing transition (except to itself with value 0).

        # Parameters
        state: The state to consider.

        # Returns
        True if the state is terminal (False otherwise).
        """
        raise NotImplementedError


class EnumerableTransitions(UncertainTransitions):
    """A domain must inherit this class if its dynamics is uncertain (with enumerable transitions) and provided as a
    white-box model.

    Compared to pure uncertain transition domains, enumerable transition ones guarantee that all probability
    distributions of next state are discrete.

    !!! tip
        Enumerable transition domains are typically stateless: they do not need to store the current state or history in
        memory since it is usually passed as parameter of their functions. By default, they only become stateful
        whenever they are used as environments (e.g. via #Initializable.reset() and #Environment.step() functions).
    """

    @autocastable
    def get_next_state_distribution(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
    ) -> DiscreteDistribution[D.T_state]:
        """Get the discrete probability distribution of next state given a memory and action.

        !!! tip
            In the Markovian case (memory only holds last state $s$), given an action $a$, this function can
            be mathematically represented by $P(S'|s, a)$, where $S'$ is the next state random variable.

        # Parameters
        memory: The source memory (state or history) of the transition.
        action: The action taken in the given memory (state or history) triggering the transition.

        # Returns
        The discrete probability distribution of next state.
        """
        return self._get_next_state_distribution(memory, action)

    def _get_next_state_distribution(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
    ) -> DiscreteDistribution[D.T_state]:
        """Get the discrete probability distribution of next state given a memory and action.

        !!! tip
            In the Markovian case (memory only holds last state $s$), given an action $a$, this function can
            be mathematically represented by $P(S'|s, a)$, where $S'$ is the next state random variable.

        # Parameters
        memory: The source memory (state or history) of the transition.
        action: The action taken in the given memory (state or history) triggering the transition.

        # Returns
        The discrete probability distribution of next state.
        """
        raise NotImplementedError


class DeterministicTransitions(EnumerableTransitions):
    """A domain must inherit this class if its dynamics is deterministic and provided as a white-box model.

    Compared to pure enumerable transition domains, deterministic transition ones guarantee that there is only one next
    state for a given source memory (state or history) and action.

    !!! tip
        Deterministic transition domains are typically stateless: they do not need to store the current state or history
        in memory since it is usually passed as parameter of their functions. By default, they only become stateful
        whenever they are used as environments (e.g. via #Initializable.reset() and #Environment.step() functions).
    """

    def _get_next_state_distribution(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
    ) -> SingleValueDistribution[D.T_state]:
        return SingleValueDistribution(self._get_next_state(memory, action))

    @autocastable
    def get_next_state(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
    ) -> D.T_state:
        """Get the next state given a memory and action.

        # Parameters
        memory: The source memory (state or history) of the transition.
        action: The action taken in the given memory (state or history) triggering the transition.

        # Returns
        The deterministic next state.
        """
        return self._get_next_state(memory, action)

    def _get_next_state(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
    ) -> D.T_state:
        """Get the next state given a memory and action.

        # Parameters
        memory: The source memory (state or history) of the transition.
        action: The action taken in the given memory (state or history) triggering the transition.

        # Returns
        The deterministic next state.
        """
        raise NotImplementedError
