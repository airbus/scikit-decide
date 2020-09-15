# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Union

from skdecide.core import StrDict

__all__ = ['MultiAgent', 'SingleAgent']


class MultiAgent:
    """A domain must inherit this class if it is multi-agent (i.e hosting multiple independent agents).

    Agents are identified by (string) agent names.
    """
    T_agent = StrDict


class SingleAgent(MultiAgent):
    """A domain must inherit this class if it is single-agent (i.e hosting only one agent)."""
    T_agent = Union
