# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Set, Union

from skdecide.core import SINGLE_AGENT_ID, StrDict

__all__ = ["MultiAgent", "SingleAgent"]


class MultiAgent:
    """A domain must inherit this class if it is multi-agent (i.e hosting multiple independent agents).

    Agents are identified by (string) agent names.
    """

    T_agent = StrDict

    def get_agents(self) -> Set[str]:
        """Return the set of available agents ids."""
        return set(self.get_observation_space())


class SingleAgent(MultiAgent):
    """A domain must inherit this class if it is single-agent (i.e hosting only one agent)."""

    T_agent = Union

    def get_agents(self) -> Set[str]:
        """Return a singleton for single agent domains.

        We must be here consistent with `skdecide.core.autocast()` which transforms a single agent domain
        into a multi agents domain whose only agent has the id "agent".

        """
        return {SINGLE_AGENT_ID}
