import functools
from copy import deepcopy
from typing import Generic, Optional, Union

from airlaps.core import T_state, T_observation, T_event, T_value, T_info, Distribution, DiscreteDistribution, \
    SingleValueDistribution, TransitionValue, EnvironmentOutcome, TransitionOutcome, Memory

__all__ = ['EnvironmentDomain', 'SimulationDomain', 'UncertainTransitionDomain', 'EnumerableTransitionDomain',
           'DeterministicTransitionDomain']


class EnvironmentDomain(Generic[T_observation, T_event, T_value, T_info]):
    """A domain must inherit this class if agents interact with it like a black-box environment.

    Black-box environment examples include: the real world, compiled ATARI games, etc.

    !!! note
        Environment domains are typically stateful: they must keep the current state or history in their memory to
        compute next steps (automatically done by default in the #_memory attribute).
    """

    def step(self, event: T_event) -> EnvironmentOutcome[T_observation, T_value, T_info]:
        """Run one step of the environment's dynamics.

        By default, #EnvironmentDomain.step() provides some boilerplate code and internally
        calls #EnvironmentDomain._step() (which returns a transition outcome). The boilerplate code automatically stores
        next state into the #_memory attribute and samples a corresponding observation.

        !!! note
            Whenever an existing environment needs to be wrapped instead of implemented fully in AIRLAPS (e.g. compiled
            ATARI games), it is recommended to overwrite #EnvironmentDomain.step() to call the external environment and
            not use the #EnvironmentDomain._step() helper function.

        !!! warning
            Before calling #EnvironmentDomain.step() the first time or when the end of an episode is
            reached, #InitializableDomain.reset() must be called to reset the environment's state.

        # Parameters
        event: The event taken in the current memory (state or history) triggering the transition.

        # Returns
        The environment outcome of this step.
        """
        transition_outcome = self._step(event)
        next_state = transition_outcome.state
        observation = self.get_observation_distribution(next_state, event).sample()
        if self._get_memory_maxlen() > 0:
            self._memory.append(deepcopy(next_state))
        return EnvironmentOutcome(observation, transition_outcome.value, transition_outcome.termination,
                                  transition_outcome.info)

    def _step(self, event: T_event) -> TransitionOutcome[T_state, T_value, T_info]:
        """Compute one step of the transition's dynamics.

        This is a helper function called by default from #EnvironmentDomain.step(). It focuses on the state level, as
        opposed to the observation one for the latter.

        # Parameters
        event: The event taken in the current memory (state or history) triggering the transition.

        # Returns
        The transition outcome of this step.
        """
        raise NotImplementedError


class SimulationDomain(EnvironmentDomain[T_observation, T_event, T_value, T_info],
                       Generic[T_observation, T_event, T_value, T_info]):
    """A domain must inherit this class if agents interact with it like a simulation.

    Compared to pure environment domains, simulation ones have the additional ability to sample transitions from any
    given state.

    !!! note
        Simulation domains are typically stateless: they do not need to store the current state or history in memory
        since it is usually passed as parameter of their functions. By default, they only become stateful whenever they
        are used as environments (e.g. via #InitializableDomain.reset() and #EnvironmentDomain.step() functions).
    """

    def _step(self, event: T_event) -> TransitionOutcome[T_state, T_value, T_info]:
        return self._sample(self._memory, event)

    def set_memory(self, memory: Union[Memory[T_state], T_state]):
        """Set internal memory attribute #_memory to given one.

        This can be useful to set a specific "starting point" before doing a rollout with
        successive #EnvironmentDomain.step() calls.

        !!! tip
            If a state is passed as memory parameter, the boilerplate code will automatically wrap it in a Memory first
            (initialized according to the domain's memory characteristic).

        # Parameters
        memory: The memory to set internally.

        # Example:
        ```python
        # Set simulation_domain memory to my_state
        simulation_domain.set_memory(my_state)

        # Start a 100-steps rollout from here (applying my_action at every step)
        for _ in range(100):
            simulation_domain.step(my_action)
        ```
        """
        if type(memory) is not Memory:
            memory = self._init_memory([deepcopy(memory)])
        self._memory = memory

    def sample(self, memory: Union[Memory[T_state], T_state], event: T_event) -> EnvironmentOutcome[
            T_observation, T_value, T_info]:
        """Sample one transition of the simulator's dynamics.

        By default, #SimulationDomain.sample() provides some boilerplate code and internally
        calls #SimulationDomain._sample() (which returns a transition outcome). The boilerplate code automatically
        samples an observation corresponding to the sampled next state.

        !!! note
            Whenever an existing simulator needs to be wrapped instead of implemented fully in AIRLAPS (e.g. a
            simulator), it is recommended to overwrite #SimulationDomain.sample() to call the external simulator and
            not use the #SimulationDomain._sample() helper function.

        !!! tip
            If a state is passed as memory parameter, the boilerplate code will automatically wrap it in a Memory first
            (initialized according to the domain's memory characteristic).

        # Parameters
        memory: The source memory (state or history) of the transition.
        event: The event taken in the given memory (state or history) triggering the transition.

        # Returns
        The environment outcome of the sampled transition.
        """
        if type(memory) is not Memory:
            memory = self._init_memory([memory])
        transition_outcome = self._sample(memory, event)
        next_state = transition_outcome.state
        observation = self.get_observation_distribution(next_state, event).sample()
        return EnvironmentOutcome(observation, transition_outcome.value, transition_outcome.termination,
                                  transition_outcome.info)

    def _sample(self, memory: Memory[T_state], event: T_event) -> TransitionOutcome[T_state, T_value, T_info]:
        """Compute one sample of the transition's dynamics.

        This is a helper function called by default from #SimulationDomain.sample(). It focuses on the state level, as
        opposed to the observation one for the latter.

        # Parameters
        memory: The source memory (state or history) of the transition.
        event: The event taken in the given memory (state or history) triggering the transition.

        # Returns
        The transition outcome of the sampled transition.
        """
        raise NotImplementedError


class UncertainTransitionDomain(SimulationDomain[T_observation, T_event, T_value, T_info],
                                Generic[T_state, T_observation, T_event, T_value, T_info]):
    """A domain must inherit this class if its dynamics is uncertain and provided as a white-box model.

    Compared to pure simulation domains, uncertain transition ones provide in addition the full probability distribution
    of next states given a memory and event.

    !!! note
        Uncertain transition domains are typically stateless: they do not need to store the current state or history in
        memory since it is usually passed as parameter of their functions. By default, they only become stateful
        whenever they are used as environments (e.g. via #InitializableDomain.reset() and #EnvironmentDomain.step()
        functions).
    """

    def _sample(self, memory: Memory[T_state], event: T_event) -> TransitionOutcome[T_state, T_value, T_info]:
        next_state = self.get_next_state_distribution(memory, event).sample()
        value = self.get_transition_value(memory, event, next_state)
        # Termination could be inferred using get_next_state_distribution based on next_state,
        # but would introduce multiple constraints on class definitions
        termination = self.is_terminal(next_state)
        return TransitionOutcome(next_state, value, termination, None)

    def get_next_state_distribution(self, memory: Union[Memory[T_state], T_state], event: T_event) -> Distribution[
            T_state]:
        """Get the probability distribution of next state given a memory and event.

        By default, #UncertainTransitionDomain.get_next_state_distribution() provides some boilerplate code and
        internally calls #UncertainTransitionDomain._get_next_state_distribution().

        !!! tip
            If a state is passed as memory parameter, the boilerplate code will automatically wrap it in a Memory first
            (initialized according to the domain's memory characteristic).

        # Parameters
        memory: The source memory (state or history) of the transition.
        event: The event taken in the given memory (state or history) triggering the transition.

        # Returns
        The probability distribution of next state.
        """
        if type(memory) is not Memory:
            memory = self._init_memory([memory])
        return self._get_next_state_distribution(memory, event)

    def _get_next_state_distribution(self, memory: Memory[T_state], event: T_event) -> Distribution[T_state]:
        """Get the probability distribution of next state given a memory and event.

        This is a helper function called by default from #UncertainTransitionDomain.get_next_state_distribution(), the
        difference being that the memory parameter is guaranteed to be of type Memory here.

        # Parameters
        memory: The source memory (state or history) of the transition.
        event: The event taken in the given memory (state or history) triggering the transition.

        # Returns
        The probability distribution of next state.
        """
        raise NotImplementedError

    def get_transition_value(self, memory: Union[Memory[T_state], T_state], event: T_event,
                             next_state: Optional[T_state] = None) -> TransitionValue[T_value]:
        """Get the value (reward or cost) of a transition.

        The transition to consider is defined by the function parameters. By
        default, #UncertainTransitionDomain.get_transition_value() provides some boilerplate code and internally
        calls #UncertainTransitionDomain._get_transition_value().

        !!! tip
            If this function never depends on the next_state parameter for its computation, it is recommended to
            indicate it by overriding #UncertainTransitionDomain._is_transition_value_dependent_on_next_state_() to
            return False. This information can then be exploited by solvers to avoid computing next state to evaluate a
            transition value (more efficient).

        !!! tip
            If a state is passed as memory parameter, the boilerplate code will automatically wrap it in a Memory first
            (initialized according to the domain's memory characteristic).
        
        # Parameters
        memory: The source memory (state or history) of the transition.
        event: The event taken in the given memory (state or history) triggering the transition.
        next_state: The next state in which the transition ends (if needed for the computation).
        
        # Returns
        The transition value (reward or cost).
        """
        if type(memory) is not Memory:
            memory = self._init_memory([memory])
        return self._get_transition_value(memory, event, next_state)

    def _get_transition_value(self, memory: Memory[T_state], event: T_event, next_state: Optional[T_state] = None) -> \
            TransitionValue[T_value]:
        """Get the value (reward or cost) of a transition.

        The transition to consider is defined by the function parameters. This is a helper function called by default
        from #UncertainTransitionDomain.get_transition_value(), the difference being that the memory parameter is
        guaranteed to be of type Memory here.

        !!! tip
            If this function never depends on the next_state parameter for its computation, it is recommended to
            indicate it by overriding #UncertainTransitionDomain._is_transition_value_dependent_on_next_state_() to
            return False. This information can then be exploited by solvers to avoid computing next state to evaluate a
            transition value (more efficient).

        # Parameters
        memory: The source memory (state or history) of the transition.
        event: The event taken in the given memory (state or history) triggering the transition.
        next_state: The next state in which the transition ends (if needed for the computation).

        # Returns
        The transition value (reward or cost).
        """
        raise NotImplementedError

    @functools.lru_cache()
    def is_transition_value_dependent_on_next_state(self) -> bool:
        """Indicate whether get_transition_value() requires the next_state parameter for its computation (cached).

        By default, #UncertainTransitionDomain.is_transition_value_dependent_on_next_state() internally
        calls #UncertainTransitionDomain._is_transition_value_dependent_on_next_state_() the first time and
        automatically caches its value to make future calls more efficient (since the returned value is assumed to be
        constant).

        # Returns
        True if the transition value computation depends on next_state (False otherwise).
        """
        return self._is_transition_value_dependent_on_next_state_()

    def _is_transition_value_dependent_on_next_state_(self) -> bool:
        """Indicate whether get_transition_value() requires the next_state parameter for its computation.

        This is a helper function called by default
        from #UncertainTransitionDomain.is_transition_value_dependent_on_next_state(), the difference being that the
        result is not cached here.

        !!! tip
            The underscore at the end of this function's name is a convention to remind that its result should be
            constant.

        # Returns
        True if the transition value computation depends on next_state (False otherwise).
        """
        return True

    def is_terminal(self, state: T_state) -> bool:
        """Indicate whether a state is terminal.

        A terminal state is a state with no outgoing transition (except to itself with value 0).

        # Parameters
        state: The state to consider.

        # Returns
        True if the state is terminal (False otherwise).
        """
        raise NotImplementedError


class EnumerableTransitionDomain(UncertainTransitionDomain[T_state, T_observation, T_event, T_value, T_info]):
    """A domain must inherit this class if its dynamics is uncertain (with enumerable transitions) and provided as a
    white-box model.

    Compared to pure uncertain transition domains, enumerable transition ones guarantee that all probability
    distributions of next state are discrete.

    !!! note
        Enumerable transition domains are typically stateless: they do not need to store the current state or history in
        memory since it is usually passed as parameter of their functions. By default, they only become stateful
        whenever they are used as environments (e.g. via #InitializableDomain.reset() and #EnvironmentDomain.step()
        functions).
    """

    def _get_next_state_distribution(self, memory: Memory[T_state], event: T_event) -> DiscreteDistribution[T_state]:
        """Get the discrete probability distribution of next state given a memory and event.

        This is a helper function called by default from #UncertainTransitionDomain.get_next_state_distribution(), the
        difference being that the memory parameter is guaranteed to be of type Memory here.

        !!! note
            In the Markovian case (memory only holds last state $s$), if the event is an action $a$, this function can
            be mathematically represented by $P(S'|s, a)$, where $S'$ is the next state random variable.

        # Parameters
        memory: The source memory (state or history) of the transition.
        event: The event taken in the given memory (state or history) triggering the transition.

        # Returns
        The discrete probability distribution of next state.
        """
        raise NotImplementedError


class DeterministicTransitionDomain(EnumerableTransitionDomain[T_state, T_observation, T_event, T_value, T_info]):
    """A domain must inherit this class if its dynamics is deterministic and provided as a white-box model.

    Compared to pure enumerable transition domains, deterministic transition ones guarantee that there is only one next
    state for a given source memory (state or history) and event.

    !!! note
        Deterministic transition domains are typically stateless: they do not need to store the current state or history
        in memory since it is usually passed as parameter of their functions. By default, they only become stateful
        whenever they are used as environments (e.g. via #InitializableDomain.reset() and #EnvironmentDomain.step()
        functions).
    """

    def _get_next_state_distribution(self, memory: Memory[T_state], event: T_event) -> SingleValueDistribution[T_state]:
        return SingleValueDistribution(self.get_next_state(memory, event))

    def get_next_state(self, memory: Union[Memory[T_state], T_state], event: T_event) -> T_state:
        """Get the next state given a memory and event.

        By default, #DeterministicTransitionDomain.get_next_state() provides some boilerplate code and internally
        calls #DeterministicTransitionDomain._get_next_state().

        !!! tip
            If a state is passed as memory parameter, the boilerplate code will automatically wrap it in a Memory first
            (initialized according to the domain's memory characteristic).

        # Parameters
        memory: The source memory (state or history) of the transition.
        event: The event taken in the given memory (state or history) triggering the transition.

        # Returns
        The deterministic next state.
        """
        if type(memory) is not Memory:
            memory = self._init_memory([memory])
        return self._get_next_state(memory, event)

    def _get_next_state(self, memory: Memory[T_state], event: T_event) -> T_state:
        """Get the next state given a memory and event.

        This is a helper function called by default from #DeterministicTransitionDomain.get_next_state(), the difference
        being that the memory parameter is guaranteed to be of type Memory here.

        # Parameters
        memory: The source memory (state or history) of the transition.
        event: The event taken in the given memory (state or history) triggering the transition.

        # Returns
        The deterministic next state.
        """
        raise NotImplementedError
