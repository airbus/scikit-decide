__all__ = ['RestorableSolver']


class RestorableSolver:
    """A solver must inherit this class if its state can be saved and restored (to continue computation later on or
    reuse its solution)."""

    def save(self, path: str) -> None:
        """Save the solver state to given path.

        # Parameters
        path: The path to store the saved state.
        """
        raise NotImplementedError

    def restore(self, path: str) -> None:
        """Restore the solver state from given path.

        # Parameters
        path: The path where the solver state was saved.
        """
        raise NotImplementedError
