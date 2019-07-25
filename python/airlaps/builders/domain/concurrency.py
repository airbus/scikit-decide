from __future__ import annotations

from typing import Union, Set

__all__ = ['Parallel', 'Sequential']


class Parallel:
    """A domain must inherit this class if multiple events/actions can happen in parallel."""
    T_concurrency = Set


class Sequential(Parallel):
    """A domain must inherit this class if its events/actions are sequential (non-parallel)."""
    T_concurrency = Union