import functools
from typing import Generic, Optional, Iterable

from airlaps.core import T_state, Memory

__all__ = ['HistoryDomain', 'FiniteHistoryDomain', 'MarkovianDomain', 'MemorylessDomain']


class HistoryDomain(Generic[T_state]):
    """A domain must inherit this class if its full state history must be stored to compute its dynamics (non-Markovian
    domain)."""
    _memory: Memory[T_state]

    def _init_memory(self, iterable: Iterable[T_state] = ()) -> Memory[T_state]:
        """Initialize memory according to its specification and return it.

        This function is automatically called by #InitializableDomain.reset() to reinitialize the internal memory
        whenever the domain is used as an environment.

        # Parameters
        iterable: An iterable of states to initialize the memory with (typically a list containing just the initial
        state).

        # Returns
        The new initialized memory.
        """
        return Memory(iterable, maxlen=self._get_memory_maxlen())

    def check_memory(self, memory: Optional[Memory[T_state]] = None) -> bool:
        """Check that the given memory (or internal one if omitted) is compliant with its memory specification (only
        checking memory length by default).

        !!! note
            This function returns always True by default because any kind of memory length should be accepted at this
            level.

        # Parameters
        memory: The memory to consider.

        # Returns
        True if the memory is compliant (False otherwise).
        """
        return True

    def get_last_state(self, memory: Optional[Memory[T_state]] = None) -> T_state:
        """Get the last state of the given memory (or internal one if omitted).

        !!! tip
            The last state of a memory can also be accessed in the Python standard way:
            ```python
            memory[-1]
            ```

        # Parameters
        memory: The memory to consider (if None, the internal memory attribute #_memory is used instead).

        # Returns
        The memory's last state.
        """
        if memory is None:
            memory = self._memory
        return memory[-1] if len(memory) > 0 else None

    def _get_memory_maxlen(self) -> Optional[int]:
        """Get the memory max length (or None if unbounded).

        !!! note
            This function returns always None by default because the memory length is unbounded at this level.

        # Returns
        The memory max length (or None if unbounded).
        """
        return None


class FiniteHistoryDomain(HistoryDomain[T_state]):
    """A domain must inherit this class if the last N states must be stored to compute its dynamics (Markovian
    domain of order N).

    N is specified by the return value of the #FiniteHistoryDomain._get_memory_maxlen() function.
    """

    def check_memory(self, memory: Optional[Memory[T_state]] = None) -> bool:
        """Check that the given memory (or internal one if omitted) is compliant with its memory specification (only
        checking memory length by default).

        # Parameters
        memory: The memory to consider (if None, the internal memory attribute #_memory is used instead).

        # Returns
        True if the memory is compliant (False otherwise).
        """
        if memory is None:
            memory = self._memory
        return memory.maxlen == self._get_memory_maxlen()

    @functools.lru_cache()
    def _get_memory_maxlen(self) -> int:
        """Get the (cached) memory max length.

        By default, #FiniteHistoryDomain._get_memory_maxlen() internally
        calls #FiniteHistoryDomain._get_memory_maxlen_() the first time and automatically caches its value to
        make future calls more efficient (since the memory max length is assumed to be constant).

        # Returns
        The memory max length.
        """
        return self._get_memory_maxlen_()

    def _get_memory_maxlen_(self) -> int:
        """Get the memory max length.

        This is a helper function called by default from #FiniteHistoryDomain._get_memory_maxlen(), the
        difference being that the result is not cached here.

        !!! tip
            The underscore at the end of this function's name is a convention to remind that its result should be
            constant.

        # Returns
        The memory max length.
        """
        raise NotImplementedError


class MarkovianDomain(FiniteHistoryDomain[T_state]):
    """A domain must inherit this class if only its last state must be stored to compute its dynamics (pure Markovian
    domain)."""

    def _get_memory_maxlen(self) -> int:
        return 1

    def _get_memory_maxlen_(self) -> int:
        pass


class MemorylessDomain(MarkovianDomain[T_state]):
    """A domain must inherit this class if it does not require any previous state(s) to be stored to compute its
    dynamics.

    A dice roll simulator is an example of memoryless domain (next states are independent of previous ones).

    !!! note
        Whenever an existing domain (environment, simulator...) needs to be wrapped instead of implemented fully in
        AIRLAPS (e.g. compiled ATARI games), the MemorylessDomain can be used because the domain memory (if any) would
        be handled externally.
    """

    def _get_memory_maxlen(self) -> int:
        return 0
