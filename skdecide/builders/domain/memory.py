# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import functools
from typing import Optional, Union

from skdecide.core import D, Memory

__all__ = ["History", "FiniteHistory", "Markovian", "Memoryless"]


class History:
    """A domain must inherit this class if its full state history must be stored to compute its dynamics (non-Markovian
    domain)."""

    _memory: D.T_memory[D.T_state]
    T_memory = Memory

    def _init_memory(self, state: Optional[D.T_state] = None) -> D.T_memory[D.T_state]:
        """Initialize memory (possibly with a state) according to its specification and return it.

        This function is automatically called by #Initializable._reset() to reinitialize the internal memory whenever
        the domain is used as an environment.

        # Parameters
        state: An optional state to initialize the memory with (typically the initial state).

        # Returns
        The new initialized memory.
        """
        content = [state] if state is not None else []
        return Memory(content, maxlen=self._get_memory_maxlen())

    def _get_memory_maxlen(self) -> Optional[int]:
        """Get the memory max length (or None if unbounded).

        !!! tip
            This function returns always None by default because the memory length is unbounded at this level.

        # Returns
        The memory max length (or None if unbounded).
        """
        return None


class FiniteHistory(History):
    """A domain must inherit this class if the last N states must be stored to compute its dynamics (Markovian
    domain of order N).

    N is specified by the return value of the #FiniteHistory._get_memory_maxlen() function.
    """

    T_memory = Memory

    @functools.lru_cache()
    def _get_memory_maxlen(self) -> int:
        """Get the (cached) memory max length.

        By default, #FiniteHistory._get_memory_maxlen() internally calls #FiniteHistory._get_memory_maxlen_() the first
        time and automatically caches its value to make future calls more efficient (since the memory max length is
        assumed to be constant).

        # Returns
        The memory max length.
        """
        return self._get_memory_maxlen_()

    def _get_memory_maxlen_(self) -> int:
        """Get the memory max length.

        This is a helper function called by default from #FiniteHistory._get_memory_maxlen(), the difference being that
        the result is not cached here.

        !!! tip
            The underscore at the end of this function's name is a convention to remind that its result should be
            constant.

        # Returns
        The memory max length.
        """
        raise NotImplementedError


class Markovian(FiniteHistory):
    """A domain must inherit this class if only its last state must be stored to compute its dynamics (pure Markovian
    domain)."""

    T_memory = Union

    def _init_memory(self, state: Optional[D.T_state] = None) -> D.T_memory[D.T_state]:
        return state

    def _get_memory_maxlen_(self) -> int:
        return 1


class Memoryless(Markovian):
    """A domain must inherit this class if it does not require any previous state(s) to be stored to compute its
    dynamics.

    A dice roll simulator is an example of memoryless domain (next states are independent of previous ones).

    !!! tip
        Whenever an existing domain (environment, simulator...) needs to be wrapped instead of implemented fully in
        scikit-decide (e.g. compiled ATARI games), Memoryless can be used because the domain memory (if any) would
        be handled externally.
    """

    T_memory = Union

    def _init_memory(self, state: Optional[D.T_state] = None) -> D.T_memory[D.T_state]:
        return None

    def _get_memory_maxlen_(self) -> int:
        return 0
