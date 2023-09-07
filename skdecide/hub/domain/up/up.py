# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Union

import unified_planning as up
from unified_planning.model.state import UPState

from skdecide.core import ImplicitSpace, Space, Value
from skdecide.domains import DeterministicPlanningDomain
from skdecide.hub.space.gym import ListSpace, MultiBinarySpace
from skdecide.utils import logger

from unified_planning.model import UPState, InstantaneousAction, Problem
from unified_planning.plans import ActionInstance
from unified_planning.model.metrics import (
    MaximizeExpressionOnFinalState,
    Oversubscription,
    TemporalOversubscription,
)
from unified_planning.shortcuts import FluentExp
from unified_planning.engines.sequential_simulator import (
    UPSequentialSimulator,
    evaluate_quality_metric,
)
from unified_planning.engines.compilers.grounder import GrounderHelper
from unified_planning.exceptions import UPValueError


class SkUPState:
    def __init__(self, up_state: UPState):
        self._up_state = up_state

    @property
    def up_state(self):
        return self._up_state

    def __hash__(self):
        return hash(frozenset(self._up_state._values.items()))

    def __eq__(self, other):
        return self._up_state._values == other._up_state._values

    def __repr__(self) -> str:
        return repr(self._up_state)

    def __str__(self) -> str:
        return str(self._up_state)


class SkUPAction:
    def __init__(self, up_action: Union[InstantaneousAction, ActionInstance]):
        if not isinstance(up_action, (InstantaneousAction, ActionInstance)):
            raise RuntimeError(
                f"SkUPAction: action {up_action} must be an instance of either InstantaneousAction or ActionInstance"
            )
        self._up_action = up_action

    @property
    def up_action(self) -> InstantaneousAction:
        return (
            self._up_action
            if isinstance(self._up_action, InstantaneousAction)
            else self._up_action.action
        )

    @property
    def up_parameters(
        self,
    ) -> Union[List[up.model.parameter.Parameter], Tuple[up.model.FNode, ...]]:
        return (
            self._up_action.parameters
            if isinstance(self._up_action, InstantaneousAction)
            else self._up_action.actual_parameters
        )

    def __hash__(self):
        return hash(self.up_action)

    def __eq__(self, other):
        return self.up_action == other.up_action

    def __repr__(self) -> str:
        return repr(self._up_action)

    def __str__(self) -> str:
        return str(self._up_action)


class D(DeterministicPlanningDomain):
    T_state = SkUPState  # Type of states
    T_observation = T_state  # Type of observations
    T_event = SkUPAction  # Type of events
    T_value = float  # Type of transition values (rewards or costs)
    T_info = None  # Type of additional information in environment outcome


class UPDomain(D):
    """This class wraps Unified Planning problem as a scikit-decide domain.

    !!! warning
        Using this class requires unified_planning to be installed.
    """

    def __init__(self, problem: Problem, **params):
        """Initialize UPDomain.

        # Parameters
        problem: The Unified Planning problem (Problem) to wrap.
        params: Optional parameters to pass to the UP sequential simulator
        """
        self._problem = problem
        self._simulator = UPSequentialSimulator(
            problem, error_on_failed_checks=True, **params
        )
        try:
            self._total_cost = FluentExp(problem.fluent("total-cost"))
        except UPValueError:
            self._total_cost = None
        self._transition_costs = {}

    def _get_next_state(self, memory: D.T_state, action: D.T_event) -> D.T_state:
        if self._total_cost is not None:
            cost = memory.up_state.get_value(self._total_cost)
        state = SkUPState(self._simulator.apply(memory.up_state, action._up_action))
        if self._total_cost is not None:
            cost = state.up_state.get_value(self._total_cost) - cost
            self._transition_costs[tuple([memory, action, state])] = cost
        return state

    def _get_transition_value(
        self,
        memory: D.T_state,
        action: D.T_event,
        next_state: Optional[D.T_state] = None,
    ) -> Value[D.T_value]:
        transition = tuple([memory, action, next_state])
        if self._total_cost is not None and transition in self._transition_costs:
            return Value(cost=self._transition_costs[transition])
        elif len(self._problem.quality_metrics) > 0:
            transition_cost = 0
            for qm in self._problem.quality_metrics:
                metric = evaluate_quality_metric(
                    self._simulator,
                    qm,
                    0,
                    memory.up_state,
                    action.up_action,
                    action.up_parameters,
                    next_state.up_state,
                )
                if isinstance(
                    qm,
                    (
                        MaximizeExpressionOnFinalState,
                        Oversubscription,
                        TemporalOversubscription,
                    ),
                ):
                    transition_cost += -metric
                else:
                    transition_cost += metric
            return Value(cost=transition_cost)
        else:
            logger.warning(
                "UPDomain: requesting transition value whereas the 'total-cost' fluent or UP quality metrics are not defined will return NaN"
            )
            return Value(cost=float("Nan"))

    def _is_terminal(self, memory: D.T_state) -> D.T_predicate:
        return self._simulator.is_goal(memory.up_state)

    def _get_action_space_(self) -> Space[D.T_event]:
        return ListSpace(
            [
                SkUPAction(a)
                for a in GrounderHelper(self._problem).get_grounded_actions()
            ]
        )

    def _get_applicable_actions_from(self, memory: D.T_state) -> Space[D.T_event]:
        app_actions = []
        for action, params in self._simulator.get_applicable_actions(memory.up_state):
            app_actions.append(
                SkUPAction(self._simulator._ground_action(action, params))
            )
        return ListSpace(app_actions)

    def _get_goals_(self) -> Space[D.T_observation]:
        return ImplicitSpace(lambda s: self._is_terminal(s))

    def _get_initial_state_(self) -> D.T_state:
        return SkUPState(self._simulator.get_initial_state())

    def _get_observation_space_(self) -> Space[D.T_observation]:
        # TODO: not clear what to do here, it will depend on the algorithm
        return MultiBinarySpace(len(self._initial_state_dict))
