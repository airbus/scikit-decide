# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Callable, Dict, List

from unified_planning.engines import Engine
from unified_planning.exceptions import UPValueError
from unified_planning.shortcuts import FluentExp, SequentialSimulator

from skdecide import Solver, Value
from skdecide.builders.solver import DeterministicPolicies, Utilities
from skdecide.hub.domain.up import SkUPAction, SkUPState, UPDomain


# TODO: remove Markovian req?
class D(UPDomain):
    pass


class UPSolver(Solver, DeterministicPolicies, Utilities):
    """This class wraps a Unified Planning engine as a scikit-decide solver.

    !!! warning
        Using this class requires unified-planning[engine] to be installed.
    """

    T_domain = D

    def __init__(
        self,
        operation_mode: Engine,
        engine_params: Dict[str, Any] = {},
        **operation_mode_params,
    ) -> None:
        """Initialize UPSolver.

        # Parameters
        operation_mode: UP operation mode class.
        engine_params: The optional dict parameters to pass to the UP engine's solve method.
        operation_mode_params: The optional dict parameters to pass to the constructor of the UP operation mode object.
        """
        super().__init__()
        self._operation_mode = operation_mode
        self._operation_mode_params = operation_mode_params
        self._engine_params = engine_params
        self._plan = []
        self._policy = {}
        self._values = {}

    def _solve_domain(self, domain_factory: Callable[[], D]) -> None:
        self._domain = domain_factory()
        problem = self._domain._problem
        om_params = (
            self._operation_mode_params
            if len(self._operation_mode_params) > 0
            else {"problem_kind": problem.kind}
        )
        with self._operation_mode(**om_params) as planner:
            result = planner.solve(problem, **self._engine_params)
            self._plan = [SkUPAction(a) for a in result.plan.actions]
            plan_extractor_domain = domain_factory()
            state = plan_extractor_domain.get_initial_state()
            self._values[state] = Value(cost=0)
            plan_cost = Value(cost=0)
            state_sequence = [state]
            for ai in self._plan:
                self._policy[state] = ai
                next_state = plan_extractor_domain.get_next_state(state, ai)
                transition_cost = plan_extractor_domain.get_transition_value(
                    state, ai, next_state
                )
                plan_cost = Value(cost=plan_cost.cost + transition_cost.cost)
                self._values[next_state] = plan_cost
                state_sequence.append(next_state)
                state = next_state
            for state in state_sequence:
                self._values[state] = Value(
                    cost=plan_cost.cost - self._values[state].cost
                )

    def _get_next_action(
        self, observation: D.T_agent[D.T_observation]
    ) -> D.T_agent[D.T_concurrency[D.T_event]]:
        return self._policy[observation]

    def _is_policy_defined_for(self, observation: D.T_agent[D.T_observation]) -> bool:
        return observation in self._policy

    def _get_utility(self, observation: D.T_agent[D.T_observation]) -> D.T_value:
        return self._values[observation]

    def get_plan(self) -> List[SkUPAction]:
        return self._plan
