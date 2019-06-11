import functools
from typing import Generic, Optional, Union

from airlaps.core import T_state, T_event, Space, Memory

__all__ = ['EventDomain', 'ActionDomain', 'UnrestrictedActionDomain']


class EventDomain(Generic[T_state, T_event]):
    """A domain must inherit this class if it handles events (controllable or not not by the agents)."""

    def get_enabled_events(self, memory: Optional[Union[Memory[T_state], T_state]] = None) -> Space[T_event]:
        """Get the space (finite or infinite set) of enabled events in the given memory (state or history), or in the
        internal one if omitted.

        By default, #EventDomain.get_enabled_events() provides some boilerplate code and internally
        calls #EventDomain._get_enabled_events(). The boilerplate code automatically passes the #_memory attribute
        instead of the memory parameter whenever the latter is None.

        !!! tip
            If a state is passed as memory parameter, the boilerplate code will automatically wrap it in a Memory first
            (initialized according to the domain's memory characteristic).

        # Parameters
        memory: The memory to consider (if None, the internal memory attribute #_memory is used instead).

        # Returns
        The space of enabled events.
        """
        if memory is None:
            memory = self._memory
        elif type(memory) is not Memory:
            memory = self._init_memory([memory])
        return self._get_enabled_events(memory)

    def _get_enabled_events(self, memory: Memory[T_state]) -> Space[T_event]:
        """Get the space (finite or infinite set) of enabled events in the given memory (state or history).

        This is a helper function called by default from #EventDomain.get_enabled_events(), the difference being that
        the memory parameter is mandatory and guaranteed to be of type Memory here.

        # Parameters
        memory: The memory to consider.

        # Returns
        The space of enabled events.
        """
        raise NotImplementedError

    def is_enabled_event(self, event: T_event, memory: Optional[Union[Memory[T_state], T_state]] = None) -> bool:
        """Indicate whether an event is enabled in the given memory (state or history), or in the internal one if
        omitted.

        By default, #EventDomain.is_enabled_event() provides some boilerplate code and internally
        calls #EventDomain._is_enabled_event(). The boilerplate code automatically passes the #_memory attribute
        instead of the memory parameter whenever the latter is None.

        !!! tip
            If a state is passed as memory parameter, the boilerplate code will automatically wrap it in a Memory first
            (initialized according to the domain's memory characteristic).

        # Parameters
        memory: The memory to consider (if None, the internal memory attribute #_memory is used instead).

        # Returns
        True if the event is enabled (False otherwise).
        """
        if memory is None:
            memory = self._memory
        elif type(memory) is not Memory:
            memory = self._init_memory([memory])
        return self._is_enabled_event(event, memory)

    def _is_enabled_event(self, event: T_event, memory: Memory[T_state]) -> bool:
        """Indicate whether an event is enabled in the given memory (state or history).

        This is a helper function called by default from #EventDomain.is_enabled_event(), the difference being that the
        memory parameter is mandatory and guaranteed to be of type Memory here.

        !!! tip
            By default, this function is implemented using the #airlaps.core.Space.contains() function on the space of
            enabled events provided by #EventDomain._get_enabled_events(), but it can be overwritten for faster
            implementations.

        # Parameters
        memory: The memory to consider.

        # Returns
        True if the event is enabled (False otherwise).
        """
        return self._get_enabled_events(memory).contains(event)

    @functools.lru_cache()
    def get_action_space(self) -> Space[T_event]:
        """Get the (cached) domain action space (finite or infinite set).

        By default, #EventDomain.get_action_space() internally calls #EventDomain._get_action_space_() the first time
        and automatically caches its value to make future calls more efficient (since the action space is assumed to be
        constant).

        # Returns
        The action space.
        """
        return self._get_action_space_()

    def _get_action_space_(self) -> Space[T_event]:
        """Get the domain action space (finite or infinite set).

        This is a helper function called by default from #EventDomain.get_action_space(), the difference being that
        the result is not cached here.

        !!! tip
            The underscore at the end of this function's name is a convention to remind that its result should be
            constant.

        # Returns
        The action space.
        """
        raise NotImplementedError

    def is_action(self, event: T_event) -> bool:
        """Indicate whether an event is an action (i.e. a controllable event for the agents).

        !!! tip
            By default, this function is implemented using the #airlaps.core.Space.contains() function on the domain
            action space provided by #EventDomain.get_action_space(), but it can be overwritten for faster
            implementations.

        # Parameters
        event: The event to consider.

        # Returns
        True if the event is an action (False otherwise).
        """
        return self.get_action_space().contains(event)

    def get_applicable_actions(self, memory: Optional[Union[Memory[T_state], T_state]] = None) -> Space[T_event]:
        """Get the space (finite or infinite set) of applicable actions in the given memory (state or history), or in
        the internal one if omitted.

        By default, #EventDomain.get_applicable_actions() provides some boilerplate code and internally
        calls #EventDomain._get_applicable_actions(). The boilerplate code automatically passes the #_memory attribute
        instead of the memory parameter whenever the latter is None.

        !!! tip
            If a state is passed as memory parameter, the boilerplate code will automatically wrap it in a Memory first
            (initialized according to the domain's memory characteristic).

        # Parameters
        memory: The memory to consider (if None, the internal memory attribute #_memory is used instead).

        # Returns
        The space of applicable actions.
        """
        if memory is None:
            memory = self._memory
        elif type(memory) is not Memory:
            memory = self._init_memory([memory])
        return self._get_applicable_actions(memory)

    def _get_applicable_actions(self, memory: Memory[T_state]) -> Space[T_event]:
        """Get the space (finite or infinite set) of applicable actions in the given memory (state or history).

        This is a helper function called by default from #EventDomain.get_applicable_actions(), the difference being
        that the memory parameter is mandatory and guaranteed to be of type Memory here.

        # Parameters
        memory: The memory to consider.

        # Returns
        The space of applicable actions.
        """
        raise NotImplementedError

    def is_applicable_action(self, event: T_event, memory: Optional[Union[Memory[T_state], T_state]] = None) -> bool:
        """Indicate whether an action is applicable in the given memory (state or history), or in the internal one if
        omitted.

        By default, #EventDomain.is_applicable_action() provides some boilerplate code and internally
        calls #EventDomain._is_applicable_action(). The boilerplate code automatically passes the #_memory attribute
        instead of the memory parameter whenever the latter is None.

        !!! tip
            If a state is passed as memory parameter, the boilerplate code will automatically wrap it in a Memory first
            (initialized according to the domain's memory characteristic).

        # Parameters
        memory: The memory to consider (if None, the internal memory attribute #_memory is used instead).

        # Returns
        True if the action is applicable (False otherwise).
        """
        if memory is None:
            memory = self._memory
        elif type(memory) is not Memory:
            memory = self._init_memory([memory])
        return self._is_applicable_action(event, memory)

    def _is_applicable_action(self, event: T_event, memory: Memory[T_state]) -> bool:
        """Indicate whether an action is applicable in the given memory (state or history).

        This is a helper function called by default from #EventDomain.is_applicable_action(), the difference being that
        the memory parameter is mandatory and guaranteed to be of type Memory here.

        !!! tip
            By default, this function is implemented using the #airlaps.core.Space.contains() function on the space of
            applicable actions provided by #EventDomain._get_applicable_actions(), but it can be overwritten for faster
            implementations.

        # Parameters
        memory: The memory to consider.

        # Returns
        True if the action is applicable (False otherwise).
        """
        return self._get_applicable_actions(memory).contains(event)


class ActionDomain(EventDomain[T_state, T_event]):
    """A domain must inherit this class if it handles only actions (i.e. controllable events)."""

    def _get_enabled_events(self, memory: Memory[T_state]) -> Space[T_event]:
        # TODO: check if by definition enabled actions is indeed equal to applicable actions
        return self._get_applicable_actions(memory)


class UnrestrictedActionDomain(ActionDomain[T_state, T_event]):
    """A domain must inherit this class if it handles only actions (i.e. controllable events), which are always all
    applicable.

    !!! note
        Applicable actions are enabled controllable events.
    """

    def _get_applicable_actions(self, memory: Memory[T_state]) -> Space[T_event]:
        return self.get_action_space()
