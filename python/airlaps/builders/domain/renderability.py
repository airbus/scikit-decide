from typing import Any, Generic, Optional, Union

from airlaps.core import T_state, Memory

__all__ = ['RenderableDomain']


class RenderableDomain(Generic[T_state]):
    """A domain must inherit this class if it can be rendered with any kind of visualization."""

    def render(self, memory: Optional[Union[Memory[T_state], T_state]] = None, **kwargs: Any) -> Any:
        """Compute a visual render of the given memory (state or history), or the internal one if omitted.

        By default, #RenderableDomain.render() provides some boilerplate code and internally
        calls #RenderableDomain._render(). The boilerplate code automatically passes the #_memory attribute instead of
        the memory parameter whenever the latter is None.

        !!! tip
            If a state is passed as memory parameter, the boilerplate code will automatically wrap it in a Memory first
            (initialized according to the domain's memory characteristic).

        # Parameters
        memory: The memory to consider (if None, the internal memory attribute #_memory is used instead).

        # Returns
        A render (e.g. image) or nothing (if the function handles the display directly).
        """
        if memory is None:
            memory = self._memory
        elif type(memory) is not Memory:
            memory = self._init_memory([memory])
        return self._render(memory, **kwargs)

    def _render(self, memory: Memory[T_state], **kwargs: Any) -> Any:
        """Compute a visual render of the given memory (state or history).

        This is a helper function called by default from #RenderableDomain.render(), the difference being that the
        memory parameter is mandatory and guaranteed to be of type Memory here.

        # Parameters
        memory: The memory to consider.

        # Returns
        A render (e.g. image) or nothing (if the function handles the display directly).
        """
        raise NotImplementedError
