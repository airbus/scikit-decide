# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Callable

from skdecide.core import D, autocastable

__all__ = ["Restorable"]


class Restorable:
    """A solver must inherit this class if its state can be saved and reloaded (to continue computation later on or
    reuse its solution)."""

    @autocastable
    def save(self, path: str) -> None:
        """Save the solver state to given path.

        # Parameters
        path: The path to store the saved state.
        """
        return self._save(path)

    def _save(self, path: str) -> None:
        """Save the solver state to given path.

        # Parameters
        path: The path to store the saved state.
        """
        raise NotImplementedError

    @autocastable
    def load(self, path: str) -> None:
        """Restore the solver state from given path.

        # Parameters
        path: The path where the solver state was saved.
        """
        return self._load(path)

    def _load(self, path: str) -> None:
        """Restore the solver state from given path.

        # Parameters
        path: The path where the solver state was saved.
        """
        raise NotImplementedError
