# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import unified_planning as up
from numpy.typing import ArrayLike
from unified_planning.engines.compilers.grounder import GrounderHelper
from unified_planning.engines.sequential_simulator import (
    UPSequentialSimulator,
    evaluate_quality_metric,
)
from unified_planning.exceptions import UPValueError
from unified_planning.model import FNode, InstantaneousAction, Problem, UPState
from unified_planning.model.metrics import (
    MaximizeExpressionOnFinalState,
    Oversubscription,
    TemporalOversubscription,
)
from unified_planning.plans import ActionInstance
from unified_planning.shortcuts import Bool, FluentExp, Int, ObjectExp, Real

from skdecide.core import EmptySpace, ImplicitSpace, Space, Value
from skdecide.domains import DeterministicPlanningDomain
from skdecide.hub.space.gym import ListSpace, SetSpace
from skdecide.hub.space.gym.gym import BoxSpace, DictSpace, DiscreteSpace, GymSpace
from skdecide.utils import logger


class SkUPState:
    def __init__(self, up_state: UPState):
        self._up_state = up_state

    @property
    def up_state(self):
        return self._up_state

    def __hash__(self):
        fs = set()
        ci = self._up_state
        while ci is not None:
            fs.update(
                (fn, v)
                for fn, v in ci._values.items()
                if fn.fluent().name != "total-cost"
            )
            ci = ci._father
        return hash(frozenset(fs))

    def __eq__(self, other):
        sd = {}
        ci = self._up_state
        while ci is not None:
            sd.update(
                {
                    fn: v
                    for fn, v in ci._values.items()
                    if fn.fluent().name != "total-cost"
                }
            )
            ci = ci._father
        od = {}
        ci = other._up_state
        while ci is not None:
            od.update(
                {
                    fn: v
                    for fn, v in ci._values.items()
                    if fn.fluent().name != "total-cost"
                }
            )
            ci = ci._father
        return sd == od

    def __repr__(self) -> str:
        return repr(self._up_state)

    def __str__(self) -> str:
        return str(self._up_state)


class SkUPAction:
    def __init__(
        self,
        up_action: Union[InstantaneousAction, ActionInstance],
        ungrounded_action: InstantaneousAction = None,
        orig_params: Tuple[FNode, ...] = None,
    ):
        if not isinstance(up_action, (InstantaneousAction, ActionInstance)):
            raise RuntimeError(
                f"SkUPAction: action {up_action} must be an instance of either InstantaneousAction or ActionInstance"
            )
        self._up_action = up_action
        self._ungrounded_action = ungrounded_action
        self._orig_params = orig_params

    @property
    def up_action(self) -> InstantaneousAction:
        return (
            self._ungrounded_action
            if self._ungrounded_action is not None
            else (
                self._up_action
                if isinstance(self._up_action, InstantaneousAction)
                else self._up_action.action
            )
        )

    @property
    def up_parameters(
        self,
    ) -> Union[List[up.model.parameter.Parameter], Tuple[up.model.FNode, ...]]:
        return (
            self._orig_params
            if self._orig_params is not None
            else (
                self._up_action.parameters
                if isinstance(self._up_action, InstantaneousAction)
                else self._up_action.actual_parameters
            )
        )

    def __hash__(self):
        return (
            hash(self._up_action)
            if isinstance(self._up_action, InstantaneousAction)
            else hash(
                tuple([self._up_action.action, self._up_action.actual_parameters])
            )
        )

    def __eq__(self, other):
        return (
            self._up_action == other._up_action
            if isinstance(self._up_action, InstantaneousAction)
            else tuple([self._up_action.action, self._up_action.actual_parameters])
            == tuple([other._up_action.action, other._up_action.actual_parameters])
        )

    def __repr__(self) -> str:
        return repr(self._up_action)

    def __str__(self) -> str:
        return str(self._up_action)


class D(DeterministicPlanningDomain):
    T_state = Union[SkUPState, Dict, ArrayLike]  # Type of states
    T_observation = T_state  # Type of observations
    T_event = SkUPAction  # Type of events
    T_value = float  # Type of transition values (rewards or costs)
    T_info = None  # Type of additional information in environment outcome


class UPDomain(D):
    """This class wraps Unified Planning problem as a scikit-decide domain.

    !!! warning
        Using this class requires unified_planning to be installed.
    """

    def __init__(
        self,
        problem: Problem,
        fluent_domains: Dict[FNode, Tuple[Union[int, float], Union[int, float]]] = None,
        state_encoding: str = "native",
        action_encoding: str = "native",
        **simulator_params,
    ):
        """Initialize UPDomain.

        # Parameters
        problem: The Unified Planning problem (Problem) to wrap.
        fluent_domains: Dictionary of min and max fluent values by fluent represented as a Unified Planning's FNode (must be provided only if get_observation_space() is used)
        state_encoding: Encoding of the state (observation) which must be one of "native", "dictionary" or "vector" (warning: if action_masking is "vector" then the state automatically becomes a dictionary which separates the action masking vector from the real state as defined here)
        action_encoding: Encoding of the action which must be either "native" or "int"
        simulator_params: Optional parameters to pass to the UP sequential simulator
        """
        self._problem = problem
        # We might be in a different process from the one where the UP problem was created
        # thus we must set the global environment to the one of the UP problem
        up.environment.GLOBAL_ENVIRONMENT = self._problem._env
        self._simulator = UPSequentialSimulator(
            self._problem, error_on_failed_checks=True, **simulator_params
        )
        self._simulator_params = simulator_params
        self._grounder = GrounderHelper(self._problem)
        try:
            self._total_cost = FluentExp(self._problem.fluent("total-cost"))
        except UPValueError:
            self._total_cost = None
        self._transition_costs = {}
        self._fluent_domains = fluent_domains
        self._state_encoding = state_encoding
        self._action_encoding = action_encoding
        self._action_space = None  # not computed by default
        self._observation_space = None  # not computed by default
        self._static_fluent_values = None
        self._fnodes_variables_map = None
        self._fnodes_vars_ordering = None
        self._states_up2np = None
        self._states_np2up = None
        self._actions_up2np = None
        self._actions_np2up = None
        if self._state_encoding != "native":
            if self._state_encoding not in ["dictionary", "vector"]:
                raise RuntimeError(
                    "State encoding must be one of 'native', 'dictionary' or 'vector'"
                )
            self._init_state_encoding_()
        if self._action_encoding != "native":
            if self._action_encoding != "int":
                raise RuntimeError("Action encoding must be either 'native' or 'int'")
            self._init_action_encoding_()

    def _init_state_encoding_(self):
        def fnode_lower_bound(fn):
            if fn.fluent().type.lower_bound is not None:
                return fn.fluent().type.lower_bound
            elif self._fluent_domains is not None and fn in self._fluent_domains:
                return self._fluent_domains[fn][0]
            else:
                raise RuntimeError(
                    f"Lower bound not provided for fluent expression {fn}"
                )

        def fnode_upper_bound(fn):
            if fn.fluent().type.upper_bound is not None:
                return fn.fluent().type.upper_bound
            elif self._fluent_domains is not None and fn in self._fluent_domains:
                return self._fluent_domains[fn][1]
            else:
                raise RuntimeError(
                    f"Upper bound not provided for fluent expression {fn}"
                )

        self._fnodes_variables_map = {}
        self._fnodes_vars_ordering = []
        self._states_up2np = {}
        self._states_np2up = {}
        init_state = self._simulator.get_initial_state()
        static_fluents = self._problem.get_static_fluents()
        self._static_fluent_values = {}
        ci = init_state
        while ci is not None:
            for fn, fv in ci._values.items():
                if (
                    fn.fluent() not in static_fluents
                    and fn.fluent().name != "total-cost"
                ):
                    self._fnodes_vars_ordering.append(fn)
                    if fn.fluent().type.is_bool_type():
                        self._fnodes_variables_map[fn] = (
                            0,
                            1,
                            lambda b_fnode: int(b_fnode.constant_value()),
                            lambda i_int: Bool(bool(i_int)),
                        )
                    elif fn.fluent().type.is_int_type():
                        lb = int(fnode_lower_bound(fn))
                        ub = int(fnode_upper_bound(fn))
                        self._fnodes_variables_map[fn] = (
                            (
                                0,
                                ub - lb + 1,
                                lambda i_fnode, lb=lb: i_fnode.constant_value() - lb,
                                lambda i_int, lb=lb: Int(i_int + lb),
                            )
                            if self._state_encoding == "dictionary"
                            else (
                                lb,
                                ub,
                                lambda i_fnode: int(i_fnode.constant_value()),
                                lambda i_int: Int(i_int),
                            )
                        )
                    elif fn.fluent().type.is_real_type():
                        self._fnodes_variables_map[fn] = (
                            float(fnode_lower_bound(fn)),
                            float(fnode_upper_bound(fn)),
                            lambda x_fnode: float(x_fnode.constant_value()),
                            lambda x_float: Real(x_float),
                        )
                    elif fn.fluent().type.is_user_type():
                        co = list(self._problem.objects(fn.fluent().type))
                        o2i = {o: i for i, o in enumerate(co)}
                        self._fnodes_variables_map[fn] = (
                            0,
                            len(co),
                            lambda o_fnode, o2i=o2i: o2i[o_fnode.object()],
                            lambda i_int, i2o=co: ObjectExp(i2o[i_int]),
                        )
                    elif fn.fluent().type.is_time_type():
                        raise RuntimeError("Time types not handled by UPDomain")
                elif fn.fluent().name != "total-cost":
                    self._static_fluent_values[fn] = fv
            ci = ci._father

    def _convert_to_skup_state_(self, state):
        if self._state_encoding == "native":
            return state
        elif self._state_encoding == "dictionary":
            kstate = frozenset(state.items())
            if kstate in self._states_np2up:
                return self._states_np2up[kstate]
            else:
                values = {}
                for fn, s in state.items():
                    values[fn] = self._fnodes_variables_map[fn][3](s)
                values.update(self._static_fluent_values)
                skup_state = SkUPState(UPState(values))
                self._states_up2np[skup_state] = state
                self._states_np2up[kstate] = skup_state
                return skup_state
        elif self._state_encoding == "vector":
            kstate = tuple(state.flatten())
            if kstate in self._states_np2up:
                return self._states_np2up[kstate]
            else:
                values = {}
                for i, fn in enumerate(self._fnodes_vars_ordering):
                    values[fn] = self._fnodes_variables_map[fn][3](state.item(i))
                values.update(self._static_fluent_values)
                skup_state = SkUPState(UPState(values))
                self._states_up2np[skup_state] = state
                self._states_np2up[kstate] = skup_state
                return skup_state
        else:
            return None

    def _convert_from_skup_state_(self, skup_state: SkUPState):
        if self._state_encoding == "native":
            return skup_state
        elif self._state_encoding == "dictionary":
            if skup_state in self._states_up2np:
                return self._states_up2np[skup_state]
            else:
                state = {}
                ci = skup_state._up_state
                while ci is not None:
                    for fn, val in ci._values.items():
                        if fn in self._fnodes_variables_map:
                            state[fn] = self._fnodes_variables_map[fn][2](val)
                    ci = ci._father
                self._states_np2up[frozenset(state.items())] = skup_state
                self._states_up2np[skup_state] = state
                return state
        elif self._state_encoding == "vector":
            if skup_state in self._states_up2np:
                return self._states_up2np[skup_state]
            else:
                state = []
                any_real = False
                for fn in self._fnodes_vars_ordering:
                    ci = skup_state._up_state
                    while ci is not None:
                        if fn in ci._values:
                            state.append(
                                self._fnodes_variables_map[fn][2](ci._values[fn])
                            )
                            break
                        ci = ci._father
                    any_real = any_real or fn.fluent().type.is_real_type()
                state = np.array(state, dtype=np.float32 if any_real else np.int32)
                self._states_np2up[tuple(state.flatten())] = skup_state
                self._states_up2np[skup_state] = state
                return state
        else:
            return None

    def _init_action_encoding_(self):
        # For actions, the numpy encoding is just an int
        self._actions_np2up = [
            SkUPAction(a[2], ungrounded_action=a[0], orig_params=a[1])
            for a in self._grounder.get_grounded_actions()
            if a[2] is not None
        ]
        self._actions_up2np = {a: i for i, a in enumerate(self._actions_np2up)}

    def _convert_to_skup_action_(self, action):
        if self._action_encoding == "native":
            return action
        elif self._action_encoding == "int":
            return self._actions_np2up[int(action)]
        else:
            return None

    def _convert_from_skup_action_(self, skup_action: SkUPAction):
        if self._action_encoding == "native":
            return skup_action
        elif self._action_encoding == "int":
            return self._actions_map[skup_action]
        else:
            return None

    def _get_next_state(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
    ) -> D.T_memory[D.T_state]:
        state = self._convert_to_skup_state_(memory)
        act = self._convert_to_skup_action_(action)
        if self._total_cost is not None:
            cost = state.up_state.get_value(self._total_cost).constant_value()
        next_state = SkUPState(
            self._simulator.apply(state.up_state, act.up_action, act.up_parameters)
        )
        if self._total_cost is not None:
            cost = (
                next_state.up_state.get_value(self._total_cost).constant_value() - cost
            )
            self._transition_costs[tuple([state, act, next_state])] = cost
        next_state = self._convert_from_skup_state_(next_state)
        return next_state

    def _get_transition_value(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
        next_state: Optional[D.T_memory[D.T_state]] = None,
    ) -> D.T_agent[Value[D.T_value]]:
        if self._total_cost is not None:
            transition = tuple(
                [
                    self._convert_to_skup_state_(memory),
                    self._convert_to_skup_action_(action),
                    self._convert_to_skup_state_(next_state),
                ]
            )
            if transition in self._transition_costs:
                return Value(cost=self._transition_costs[transition])
        if len(self._problem.quality_metrics) > 0:
            transition_cost = 0
            state = self._convert_to_skup_state_(memory)
            act = self._convert_to_skup_action_(action)
            nstate = self._convert_to_skup_state_(next_state)
            for qm in self._problem.quality_metrics:
                metric = evaluate_quality_metric(
                    self._simulator,
                    qm,
                    0,
                    state.up_state,
                    act.up_action,
                    act.up_parameters,
                    nstate.up_state,
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

    def _is_terminal(self, memory: D.T_state) -> D.T_agent[D.T_predicate]:
        state = self._convert_to_skup_state_(memory)
        return self._simulator.is_goal(state.up_state)

    def _get_action_space_(self) -> D.T_agent[Space[D.T_event]]:
        if self._action_space is None:
            if self._action_encoding == "native":
                # By default we don't initialize action encoding for actions
                # since it is not highly expected that the action space will
                # be requested from the user in "native" action encoding mode
                if self._actions_np2up is None:
                    self._init_action_encoding_()
                self._action_space = ListSpace(self._actions_np2up)
            elif self._action_encoding == "int":
                self._action_space = DiscreteSpace(len(self._actions_np2up))
            else:
                return None
        return self._action_space

    def _get_applicable_actions_from(
        self, memory: D.T_state
    ) -> D.T_agent[Space[D.T_event]]:
        state = self._convert_to_skup_state_(memory)
        applicable_actions = [
            SkUPAction(
                self._grounder.ground_action(action, params),
                ungrounded_action=action,
                orig_params=params,
            )
            for action, params in self._simulator.get_applicable_actions(state.up_state)
        ]
        if self._action_encoding == "native":
            # SetSpace requires to be non empty
            return (
                SetSpace(applicable_actions)
                if len(applicable_actions) > 0
                else EmptySpace()
            )
        elif self._action_encoding == "int":
            # SetSpace requires to be non empty
            return (
                SetSpace([self._actions_up2np[a] for a in applicable_actions])
                if len(applicable_actions) > 0
                else EmptySpace()
            )
        else:
            return None

    def _get_goals_(self) -> D.T_agent[Space[D.T_observation]]:
        return ImplicitSpace(lambda s: self._is_terminal(s))

    def _get_initial_state_(self) -> D.T_state:
        init_state = self._convert_from_skup_state_(
            SkUPState(self._simulator.get_initial_state())
        )
        return init_state

    def _get_observation_space_(self) -> D.T_agent[Space[D.T_observation]]:
        if self._observation_space is None:
            if self._state_encoding == "native":
                raise RuntimeError(
                    "Observation space defined only for state encoding 'dictionary' or 'vector'"
                )
            elif self._state_encoding == "dictionary":
                self._observation_space = DictSpace(
                    {
                        repr(fn): (
                            DiscreteSpace(2)
                            if fn.fluent().type.is_bool_type()
                            else (
                                DiscreteSpace(v[1])
                                if fn.fluent().type.is_int_type()
                                else (
                                    BoxSpace(v[0], v[1])
                                    if fn.fluent().type.is_real_type()
                                    else (
                                        DiscreteSpace(v[1])
                                        if fn.fluent().type.is_user_type()
                                        else None
                                    )
                                )
                            )
                        )
                        for fn, v in self._fnodes_variables_map.items()
                    }
                )
            elif self._state_encoding == "vector":
                self._observation_space = BoxSpace(
                    low=np.array(
                        [
                            self._fnodes_variables_map[fn][0]
                            for fn in self._fnodes_vars_ordering
                        ]
                    ),
                    high=np.array(
                        [
                            self._fnodes_variables_map[fn][1]
                            for fn in self._fnodes_vars_ordering
                        ]
                    ),
                    dtype=(
                        np.float32
                        if any(
                            fn.fluent().type.is_real_type()
                            for fn in self._fnodes_vars_ordering
                        )
                        else np.int32
                    ),
                )
            else:
                return None
        return self._observation_space
