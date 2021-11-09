# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Optional

from skdecide.core import D, autocastable

__all__ = ["Renderable"]


class Renderable:
    """A domain must inherit this class if it can be rendered with any kind of visualization."""

    @autocastable
    def render(
        self, memory: Optional[D.T_memory[D.T_state]] = None, **kwargs: Any
    ) -> Any:
        """Compute a visual render of the given memory (state or history), or the internal one if omitted.

        By default, #Renderable.render() provides some boilerplate code and internally calls #Renderable._render(). The
        boilerplate code automatically passes the #_memory attribute instead of the memory parameter whenever the latter
        is None.

        # Parameters
        memory: The memory to consider (if None, the internal memory attribute #_memory is used instead).

        # Returns
        A render (e.g. image) or nothing (if the function handles the display directly).
        """
        return self._render(memory, **kwargs)

    def _render(
        self, memory: Optional[D.T_memory[D.T_state]] = None, **kwargs: Any
    ) -> Any:
        """Compute a visual render of the given memory (state or history), or the internal one if omitted.

        By default, #Renderable._render() provides some boilerplate code and internally
        calls #Renderable._render_from(). The boilerplate code automatically passes the #_memory attribute instead of
        the memory parameter whenever the latter is None.

        # Parameters
        memory: The memory to consider (if None, the internal memory attribute #_memory is used instead).

        # Returns
        A render (e.g. image) or nothing (if the function handles the display directly).
        """
        if memory is None:
            memory = self._memory
        return self._render_from(memory, **kwargs)

    def _render_from(self, memory: D.T_memory[D.T_state], **kwargs: Any) -> Any:
        """Compute a visual render of the given memory (state or history).

        This is a helper function called by default from #Renderable._render(), the difference being that the
        memory parameter is mandatory here.

        # Parameters
        memory: The memory to consider.

        # Returns
        A render (e.g. image) or nothing (if the function handles the display directly).
        """
        raise NotImplementedError
