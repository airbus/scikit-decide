from typing import Any, Generic, Optional, Callable

from airlaps.core import T_observation, Memory

__all__ = ['SolutionSolver', 'AnytimeSolver', 'RealtimeSolver']


class SolutionSolver(Generic[T_observation]):
    """A solver must inherit this class if it can solve (i.e. find solutions)."""

    # TODO: call a new _solve function where from_observation would not be optional
    #  (it would automatically be set by the solve function to the domain initial state if None)?
    def solve(self, from_observation: Optional[Memory[T_observation]] = None,
              on_update: Optional[Callable[..., bool]] = None, max_time: Optional[float] = None,
              **kwargs: Any) -> None:
        """Start solving process.

        !!! note
            The nature of the solutions produced here depends on other solver's characteristics like
            #policy and #assessibility.

        # Parameters
        from_observation: The observation memory to solve from (if None, the domain's initial one is used).
        on_update: An optional callable to call at every iteration of the solving process.
        max_time: The maximum time duration for the solver to solve (if None, infinity is assumed).
        kwargs: Any additional keyword arguments that this specific solver might need.
        """
        raise NotImplementedError


class AnytimeSolver(SolutionSolver[T_observation]):
    """A solver must inherit this class if it can solve in anytime (i.e. find solutions with intermediate valid
    solutions available as soon as discovered)."""
    pass


class RealtimeSolver(AnytimeSolver[T_observation]):
    """A solver must inherit this class if it can solve in realtime (i.e. find solutions with intermediate valid
    solutions available in less than a customizable duration)."""
    pass

# # First proposal:
# class RealtimeSolver(Generic[T_observation]):
#
#     def solve(self, from_observation: Optional[Memory[T_observation]] = None,
#               on_update: Optional[Callable[[], bool]] = None, max_time: Optional[float] = None) -> None:
#         raise NotImplementedError
#
#
# class AnytimeSolver(Generic[T_observation]):
#
#     def solve(self, from_observation: Optional[Memory[T_observation]] = None,
#               on_update: Optional[Callable[[], bool]] = None) -> None:
#         raise NotImplementedError
#
#
# class OfflineSolver(Generic[T_observation]):
#
#     def solve(self, from_observation: Optional[Memory[T_observation]] = None) -> None:
#         raise NotImplementedError
