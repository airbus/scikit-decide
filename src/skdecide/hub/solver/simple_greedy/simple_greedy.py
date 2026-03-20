# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from typing import Optional

from skdecide import DeterministicPolicySolver, Domain, EnumerableSpace, Memory
from skdecide.builders.domain import EnumerableTransitions, FullyObservable, SingleAgent
from skdecide.core import autocast

logger = logging.getLogger(__name__)


class D(Domain, SingleAgent, EnumerableTransitions, FullyObservable):
    pass


class SimpleGreedy(DeterministicPolicySolver):
    T_domain = D

    @classmethod
    def _check_domain_additional(cls, domain: D) -> bool:
        return isinstance(domain.get_action_space(), EnumerableSpace)

    def _solve(self) -> None:
        self._domain = (
            self._domain_factory()
        )  # no further solving code required here since everything is computed online

    def _get_next_action(
        self, observation: D.T_agent[D.T_observation], domain: Optional[D] = None
    ) -> D.T_agent[D.T_concurrency[D.T_event]]:
        if domain is None:
            domain = self._domain
            logger.warning(
                "Rollout domain not given. Using domain seen during solve instead."
            )
        # This solver selects the first action with the highest expected immediate reward (greedy)
        memory = Memory(
            [observation]
        )  # note: observation == state (because FullyObservable)
        get_applicable_actions = autocast(
            domain.get_applicable_actions, domain, self.T_domain
        )
        get_next_state_distribution = autocast(
            domain.get_next_state_distribution, domain, self.T_domain
        )
        get_transition_value = autocast(
            domain.get_transition_value, domain, self.T_domain
        )
        applicable_actions = get_applicable_actions(memory)
        if domain.is_transition_value_dependent_on_next_state():
            values = []
            for a in applicable_actions.get_elements():
                next_state_prob = get_next_state_distribution(memory, [a]).get_values()
                expected_value = sum(
                    p * get_transition_value(memory, [a], s).reward
                    for s, p in next_state_prob
                )
                values.append(expected_value)
        else:
            values = [
                get_transition_value(memory, a).reward for a in applicable_actions
            ]
        argmax = max(range(len(values)), key=lambda i: values[i])
        return [
            applicable_actions.get_elements()[argmax]
        ]  # list of action here because we handle Parallel domains

    def _is_policy_defined_for(self, observation: D.T_agent[D.T_observation]) -> bool:
        return True
