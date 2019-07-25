from __future__ import annotations

from typing import Union

from airlaps.core import StrDict

__all__ = ['MultiAgent', 'SingleAgent']


class MultiAgent:
    """A domain must inherit this class if it is multi-agent (i.e hosting multiple independent agents).

    Agents are identified by (string) agent names.
    """
    T_agent = StrDict


class SingleAgent(MultiAgent):
    """A domain must inherit this class if it is single-agent (i.e hosting only one agent)."""
    T_agent = Union
