# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Callable

from skdecide import DeterministicPolicySolver, Domain, EnumerableSpace, Memory
from skdecide.builders.domain import EnumerableTransitions, FullyObservable, SingleAgent


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
        self, observation: D.T_agent[D.T_observation]
    ) -> D.T_agent[D.T_concurrency[D.T_event]]:
        # This solver selects the first action with the highest expected immediate reward (greedy)
        domain = self._domain
        memory = Memory(
            [observation]
        )  # note: observation == state (because FullyObservable)
        applicable_actions = domain.get_applicable_actions(memory)
        if domain.is_transition_value_dependent_on_next_state():
            values = []
            for a in applicable_actions.get_elements():
                next_state_prob = domain.get_next_state_distribution(
                    memory, [a]
                ).get_values()
                expected_value = sum(
                    p * domain.get_transition_value(memory, [a], s).reward
                    for s, p in next_state_prob
                )
                values.append(expected_value)
        else:
            values = [
                domain.get_transition_value(memory, a).reward
                for a in applicable_actions
            ]
        argmax = max(range(len(values)), key=lambda i: values[i])
        return [
            applicable_actions.get_elements()[argmax]
        ]  # list of action here because we handle Parallel domains

    def _is_policy_defined_for(self, observation: D.T_agent[D.T_observation]) -> bool:
        return True
