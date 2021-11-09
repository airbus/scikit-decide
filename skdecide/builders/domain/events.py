# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import functools
from typing import Optional, Union

from skdecide.core import D, EmptySpace, Space, autocastable

__all__ = ["Events", "Actions", "UnrestrictedActions"]


class Events:
    """A domain must inherit this class if it handles events (controllable or not not by the agents)."""

    @autocastable
    def get_enabled_events(
        self, memory: Optional[D.T_memory[D.T_state]] = None
    ) -> Space[D.T_event]:
        """Get the space (finite or infinite set) of enabled uncontrollable events in the given memory (state or
        history), or in the internal one if omitted.

        By default, #Events.get_enabled_events() provides some boilerplate code and internally
        calls #Events._get_enabled_events(). The boilerplate code automatically passes the #_memory attribute instead of
        the memory parameter whenever the latter is None.

        # Parameters
        memory: The memory to consider (if None, the internal memory attribute #_memory is used instead).

        # Returns
        The space of enabled events.
        """
        return self._get_enabled_events(memory)

    def _get_enabled_events(
        self, memory: Optional[D.T_memory[D.T_state]] = None
    ) -> Space[D.T_event]:
        """Get the space (finite or infinite set) of enabled uncontrollable events in the given memory (state or
        history), or in the internal one if omitted.

        By default, #Events._get_enabled_events() provides some boilerplate code and internally
        calls #Events._get_enabled_events_from(). The boilerplate code automatically passes the #_memory attribute
        instead of the memory parameter whenever the latter is None.

        # Parameters
        memory: The memory to consider (if None, the internal memory attribute #_memory is used instead).

        # Returns
        The space of enabled events.
        """
        if memory is None:
            memory = self._memory
        return self._get_enabled_events_from(memory)

    def _get_enabled_events_from(
        self, memory: D.T_memory[D.T_state]
    ) -> Space[D.T_event]:
        """Get the space (finite or infinite set) of enabled uncontrollable events in the given memory (state or
        history).

        This is a helper function called by default from #Events._get_enabled_events(), the difference being that the
        memory parameter is mandatory here.

        # Parameters
        memory: The memory to consider.

        # Returns
        The space of enabled events.
        """
        raise NotImplementedError

    @autocastable
    def is_enabled_event(
        self, event: D.T_event, memory: Optional[D.T_memory[D.T_state]] = None
    ) -> bool:
        """Indicate whether an uncontrollable event is enabled in the given memory (state or history), or in the
        internal one if omitted.

        By default, #Events.is_enabled_event() provides some boilerplate code and internally
        calls #Events._is_enabled_event(). The boilerplate code automatically passes the #_memory attribute instead of
        the memory parameter whenever the latter is None.

        # Parameters
        memory: The memory to consider (if None, the internal memory attribute #_memory is used instead).

        # Returns
        True if the event is enabled (False otherwise).
        """
        return self._is_enabled_event(event, memory)

    def _is_enabled_event(
        self, event: D.T_event, memory: Optional[D.T_memory[D.T_state]] = None
    ) -> bool:
        """Indicate whether an uncontrollable event is enabled in the given memory (state or history), or in the
        internal one if omitted.

        By default, #Events._is_enabled_event() provides some boilerplate code and internally
        calls #Events._is_enabled_event_from(). The boilerplate code automatically passes the #_memory attribute instead
        of the memory parameter whenever the latter is None.

        # Parameters
        memory: The memory to consider (if None, the internal memory attribute #_memory is used instead).

        # Returns
        True if the event is enabled (False otherwise).
        """
        if memory is None:
            memory = self._memory
        return self._is_enabled_event_from(event, memory)

    def _is_enabled_event_from(
        self, event: D.T_event, memory: D.T_memory[D.T_state]
    ) -> bool:
        """Indicate whether an event is enabled in the given memory (state or history).

        This is a helper function called by default from #Events._is_enabled_event(), the difference being that the
        memory parameter is mandatory here.

        !!! tip
            By default, this function is implemented using the #skdecide.core.Space.contains() function on the space of
            enabled events provided by #Events._get_enabled_events_from(), but it can be overridden for faster
            implementations.

        # Parameters
        memory: The memory to consider.

        # Returns
        True if the event is enabled (False otherwise).
        """
        return self._get_enabled_events_from(memory).contains(event)

    @autocastable
    def get_action_space(self) -> D.T_agent[Space[D.T_event]]:
        """Get the (cached) domain action space (finite or infinite set).

        By default, #Events.get_action_space() internally calls #Events._get_action_space_() the first time and
        automatically caches its value to make future calls more efficient (since the action space is assumed to be
        constant).

        # Returns
        The action space.
        """
        return self._get_action_space()

    @functools.lru_cache()
    def _get_action_space(self) -> D.T_agent[Space[D.T_event]]:
        """Get the (cached) domain action space (finite or infinite set).

        By default, #Events._get_action_space() internally calls #Events._get_action_space_() the first time and
        automatically caches its value to make future calls more efficient (since the action space is assumed to be
        constant).

        # Returns
        The action space.
        """
        return self._get_action_space_()

    def _get_action_space_(self) -> D.T_agent[Space[D.T_event]]:
        """Get the domain action space (finite or infinite set).

        This is a helper function called by default from #Events._get_action_space(), the difference being that the
        result is not cached here.

        !!! tip
            The underscore at the end of this function's name is a convention to remind that its result should be
            constant.

        # Returns
        The action space.
        """
        raise NotImplementedError

    @autocastable
    def is_action(self, event: D.T_event) -> bool:
        """Indicate whether an event is an action (i.e. a controllable event for the agents).

        !!! tip
            By default, this function is implemented using the #skdecide.core.Space.contains() function on the domain
            action space provided by #Events.get_action_space(), but it can be overridden for faster implementations.

        # Parameters
        event: The event to consider.

        # Returns
        True if the event is an action (False otherwise).
        """
        return self._is_action(event)

    def _is_action(self, event: D.T_event) -> bool:
        """Indicate whether an event is an action (i.e. a controllable event for the agents).

        !!! tip
            By default, this function is implemented using the #skdecide.core.Space.contains() function on the domain
            action space provided by #Events._get_action_space(), but it can be overridden for faster implementations.

        # Parameters
        event: The event to consider.

        # Returns
        True if the event is an action (False otherwise).
        """
        return self._get_action_space().contains(event)

    @autocastable
    def get_applicable_actions(
        self, memory: Optional[D.T_memory[D.T_state]] = None
    ) -> D.T_agent[Space[D.T_event]]:
        """Get the space (finite or infinite set) of applicable actions in the given memory (state or history), or in
        the internal one if omitted.

        By default, #Events.get_applicable_actions() provides some boilerplate code and internally
        calls #Events._get_applicable_actions(). The boilerplate code automatically passes the #_memory attribute
        instead of the memory parameter whenever the latter is None.

        # Parameters
        memory: The memory to consider (if None, the internal memory attribute #_memory is used instead).

        # Returns
        The space of applicable actions.
        """
        return self._get_applicable_actions(memory)

    def _get_applicable_actions(
        self, memory: Optional[D.T_memory[D.T_state]] = None
    ) -> D.T_agent[Space[D.T_event]]:
        """Get the space (finite or infinite set) of applicable actions in the given memory (state or history), or in
        the internal one if omitted.

        By default, #Events._get_applicable_actions() provides some boilerplate code and internally
        calls #Events._get_applicable_actions_from(). The boilerplate code automatically passes the #_memory attribute
        instead of the memory parameter whenever the latter is None.

        # Parameters
        memory: The memory to consider (if None, the internal memory attribute #_memory is used instead).

        # Returns
        The space of applicable actions.
        """
        if memory is None:
            memory = self._memory
        return self._get_applicable_actions_from(memory)

    def _get_applicable_actions_from(
        self, memory: D.T_memory[D.T_state]
    ) -> D.T_agent[Space[D.T_event]]:
        """Get the space (finite or infinite set) of applicable actions in the given memory (state or history).

        This is a helper function called by default from #Events._get_applicable_actions(), the difference being that
        the memory parameter is mandatory here.

        # Parameters
        memory: The memory to consider.

        # Returns
        The space of applicable actions.
        """
        raise NotImplementedError

    @autocastable
    def is_applicable_action(
        self,
        action: D.T_agent[D.T_event],
        memory: Optional[D.T_memory[D.T_state]] = None,
    ) -> bool:
        """Indicate whether an action is applicable in the given memory (state or history), or in the internal one if
        omitted.

        By default, #Events.is_applicable_action() provides some boilerplate code and internally
        calls #Events._is_applicable_action(). The boilerplate code automatically passes the #_memory attribute instead
        of the memory parameter whenever the latter is None.

        # Parameters
        memory: The memory to consider (if None, the internal memory attribute #_memory is used instead).

        # Returns
        True if the action is applicable (False otherwise).
        """
        return self._is_applicable_action(action, memory)

    def _is_applicable_action(
        self,
        action: D.T_agent[D.T_event],
        memory: Optional[D.T_memory[D.T_state]] = None,
    ) -> bool:
        """Indicate whether an action is applicable in the given memory (state or history), or in the internal one if
        omitted.

        By default, #Events._is_applicable_action() provides some boilerplate code and internally
        calls #Events._is_applicable_action_from(). The boilerplate code automatically passes the #_memory attribute
        instead of the memory parameter whenever the latter is None.

        # Parameters
        memory: The memory to consider (if None, the internal memory attribute #_memory is used instead).

        # Returns
        True if the action is applicable (False otherwise).
        """
        if memory is None:
            memory = self._memory
        return self._is_applicable_action_from(action, memory)

    def _is_applicable_action_from(
        self, action: D.T_agent[D.T_event], memory: D.T_memory[D.T_state]
    ) -> bool:
        """Indicate whether an action is applicable in the given memory (state or history).

        This is a helper function called by default from #Events._is_applicable_action(), the difference being that the
        memory parameter is mandatory here.

        !!! tip
            By default, this function is implemented using the #skdecide.core.Space.contains() function on the space of
            applicable actions provided by #Events._get_applicable_actions_from(), but it can be overridden for faster
            implementations.

        # Parameters
        memory: The memory to consider.

        # Returns
        True if the action is applicable (False otherwise).
        """
        applicable_actions = self._get_applicable_actions_from(memory)
        if self.T_agent == Union:
            return applicable_actions.contains(action)
        else:  # StrDict
            return all(applicable_actions[k].contains(v) for k, v in action.items())


class Actions(Events):
    """A domain must inherit this class if it handles only actions (i.e. controllable events)."""

    def _get_enabled_events_from(
        self, memory: D.T_memory[D.T_state]
    ) -> Space[D.T_event]:
        # TODO: check definition of enabled events (only uncontrollable?)
        # return self._get_enabled_actions_from(memory)
        return EmptySpace()


class UnrestrictedActions(Actions):
    """A domain must inherit this class if it handles only actions (i.e. controllable events), which are always all
    applicable.
    """

    def _get_applicable_actions_from(
        self, memory: D.T_memory[D.T_state]
    ) -> D.T_agent[Space[D.T_event]]:
        return self._get_action_space()
