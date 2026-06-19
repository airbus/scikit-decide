# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Optional

from skdecide import (
    DeterministicPlanningDomain,
    DiscreteDistribution,
    GoalMDPDomain,
    ImplicitSpace,
    Space,
    Value,
)
from skdecide.builders.domain import UnrestrictedActions
from skdecide.core import D
from skdecide.hub.domain.pddl.pddl import PDDLReader
from skdecide.hub.space.gym import ListSpace

try:
    from skdecide.hub.__skdecide_hub_cpp import (
        _PDDL_ApplicableActionsGenerator_ as CppApplicableActionsGenerator,
    )
    from skdecide.hub.__skdecide_hub_cpp import (
        _PDDL_GoalChecker_ as CppGoalChecker,
    )
    from skdecide.hub.__skdecide_hub_cpp import (
        _PDDL_SuccessorGenerator_ as CppSuccessorGenerator,
    )
    from skdecide.hub.__skdecide_hub_cpp import _PDDL_Task_ as CppTask
    from skdecide.hub.__skdecide_hub_cpp import (
        _PDDL_TemporalSimulator_ as CppTemporalSimulator,
    )
except ImportError:
    print(
        "Scikit-decide C++ hub library not found. "
        'Please check it is installed in "skdecide/hub".'
    )
    raise

try:
    import z3 as _z3_mod  # noqa: F401

    _HAS_Z3_PYTHON = True
except Exception:
    _HAS_Z3_PYTHON = False


class PDDLState:
    """Immutable, hashable PDDL state wrapping the C++ State."""

    def __init__(self, cpp_state, cost_function_ids=frozenset()):
        self._cpp_state = cpp_state
        self._atoms = tuple(
            frozenset(tuple(t) for t in s) for s in cpp_state.get_atoms()
        )
        fluents_raw = cpp_state.get_fluents()
        self._fluents = tuple(
            frozenset((tuple(k), v) for k, v in f.items())
            if i not in cost_function_ids
            else frozenset()
            for i, f in enumerate(fluents_raw)
        )
        self._hash = hash((self._atoms, self._fluents))

    def to_cpp(self):
        return self._cpp_state

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return (
            isinstance(other, PDDLState)
            and self._atoms == other._atoms
            and self._fluents == other._fluents
        )

    def __repr__(self):
        return f"PDDLState(atoms={self._atoms})"


class PDDLAction:
    """Hashable ground action wrapper."""

    def __init__(self, cpp_ground_action, task):
        self._cpp_action = cpp_ground_action
        self.action_id = cpp_ground_action.action_id
        self.arguments = tuple(cpp_ground_action.arguments)
        self._task = task

    def to_cpp(self):
        return self._cpp_action

    def __hash__(self):
        return hash((self.action_id, self.arguments))

    def __eq__(self, other):
        return (
            isinstance(other, PDDLAction)
            and self.action_id == other.action_id
            and self.arguments == other.arguments
        )

    def __repr__(self):
        name = self._task.action_name(self.action_id)
        args = " ".join(self._task.object_name(a) for a in self.arguments)
        if args:
            return f"({name} {args})"
        return f"({name})"


class PDDLDomain(DeterministicPlanningDomain, UnrestrictedActions):
    """Deterministic PDDL 2.1 planning domain backed by a C++/clingo engine.

    # Parameters
    domain_path: Path to the PDDL domain file.
    problem_path: Path to the PDDL problem file.
    """

    T_state = PDDLState
    T_observation = T_state
    T_event = PDDLAction

    def __init__(self, domain_path: str, problem_path: str):
        self._reader = PDDLReader(domain_path, problem_path)
        self._task = CppTask(self._reader.domains[0], self._reader.problems[0])
        self._aops_gen = CppApplicableActionsGenerator(self._task)
        self._succ_gen = CppSuccessorGenerator(self._task)
        self._goal_checker = CppGoalChecker(self._task)

        self._cost_function_ids = set()
        tc = self._task.total_cost_function()
        if tc >= 0:
            self._cost_function_ids.add(tc)
            self._total_cost_idx = tc
        else:
            self._total_cost_idx = -1
        rw = self._task.reward_function()
        if rw >= 0:
            self._cost_function_ids.add(rw)
            self._reward_idx = rw
        else:
            self._reward_idx = -1
        self._cost_function_ids = frozenset(self._cost_function_ids)

    def _get_initial_state_(self) -> D.T_state:
        return PDDLState(self._task.initial_state(), self._cost_function_ids)

    def _get_next_state(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
    ) -> D.T_state:
        succs = self._succ_gen.get_successors(memory.to_cpp(), action.to_cpp())
        assert len(succs) == 1, f"Expected 1 deterministic successor, got {len(succs)}"
        return PDDLState(succs[0].state, self._cost_function_ids)

    def _get_transition_value(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
        next_state: Optional[D.T_state] = None,
    ) -> D.T_agent[Value[D.T_value]]:
        if self._total_cost_idx >= 0 and next_state is not None:
            ns_fluents = next_state.to_cpp().get_fluents()
            ms_fluents = memory.to_cpp().get_fluents()
            if self._total_cost_idx < len(ns_fluents):
                ns_tc = ns_fluents[self._total_cost_idx]
                ms_tc = ms_fluents[self._total_cost_idx]
                ns_val = ns_tc.get((), 0.0)
                ms_val = ms_tc.get((), 0.0)
                return Value(cost=ns_val - ms_val)
        return Value(cost=1.0)

    def _get_applicable_actions_from(
        self, memory: D.T_memory[D.T_state]
    ) -> D.T_agent[Space[D.T_event]]:
        cpp_actions = self._aops_gen.get_applicable_actions(memory.to_cpp())
        return ListSpace([PDDLAction(a, self._task) for a in cpp_actions])

    def _get_goals_(self) -> D.T_agent[Space[D.T_observation]]:
        return ImplicitSpace(lambda s: self._goal_checker.is_goal(s.to_cpp()))

    def _is_terminal(self, state: D.T_state) -> D.T_agent[D.T_predicate]:
        if self._goal_checker.is_goal(state.to_cpp()):
            return True
        actions = self._aops_gen.get_applicable_actions(state.to_cpp())
        return len(actions) == 0

    def _get_action_space_(self) -> D.T_agent[Space[D.T_event]]:
        return ImplicitSpace(lambda a: isinstance(a, PDDLAction))

    def _get_observation_space_(self) -> D.T_agent[Space[D.T_observation]]:
        return ImplicitSpace(lambda s: isinstance(s, PDDLState))

    @property
    def task(self):
        return self._task

    @property
    def cpp_domain(self):
        return self._reader.domains[0]

    @property
    def cpp_problem(self):
        return self._reader.problems[0]


class PPDDLDomain(GoalMDPDomain, UnrestrictedActions):
    """Probabilistic PPDDL planning domain backed by a C++/clingo engine.

    # Parameters
    domain_path: Path to the PPDDL domain file.
    problem_path: Path to the PPDDL problem file.
    """

    T_state = PDDLState
    T_observation = T_state
    T_event = PDDLAction

    def __init__(self, domain_path: str, problem_path: str):
        self._reader = PDDLReader(domain_path, problem_path)
        self._task = CppTask(self._reader.domains[0], self._reader.problems[0])
        self._aops_gen = CppApplicableActionsGenerator(self._task)
        self._succ_gen = CppSuccessorGenerator(self._task)
        self._goal_checker = CppGoalChecker(self._task)

        self._cost_function_ids = set()
        tc = self._task.total_cost_function()
        if tc >= 0:
            self._cost_function_ids.add(tc)
            self._total_cost_idx = tc
        else:
            self._total_cost_idx = -1
        rw = self._task.reward_function()
        if rw >= 0:
            self._cost_function_ids.add(rw)
            self._reward_idx = rw
        else:
            self._reward_idx = -1
        self._cost_function_ids = frozenset(self._cost_function_ids)

    def _get_initial_state_(self) -> D.T_state:
        return PDDLState(self._task.initial_state(), self._cost_function_ids)

    def _get_next_state_distribution(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
    ) -> DiscreteDistribution[D.T_state]:
        succs = self._succ_gen.get_successors(memory.to_cpp(), action.to_cpp())
        return DiscreteDistribution(
            [
                (PDDLState(s.state, self._cost_function_ids), s.probability)
                for s in succs
            ]
        )

    def _get_transition_value(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
        next_state: Optional[D.T_state] = None,
    ) -> D.T_agent[Value[D.T_value]]:
        if self._reward_idx >= 0 and next_state is not None:
            ns_fluents = next_state.to_cpp().get_fluents()
            ms_fluents = memory.to_cpp().get_fluents()
            if self._reward_idx < len(ns_fluents):
                ns_rw = ns_fluents[self._reward_idx]
                ms_rw = ms_fluents[self._reward_idx]
                ns_val = ns_rw.get((), 0.0)
                ms_val = ms_rw.get((), 0.0)
                return Value(cost=-(ns_val - ms_val))
        if self._total_cost_idx >= 0 and next_state is not None:
            ns_fluents = next_state.to_cpp().get_fluents()
            ms_fluents = memory.to_cpp().get_fluents()
            if self._total_cost_idx < len(ns_fluents):
                ns_tc = ns_fluents[self._total_cost_idx]
                ms_tc = ms_fluents[self._total_cost_idx]
                ns_val = ns_tc.get((), 0.0)
                ms_val = ms_tc.get((), 0.0)
                return Value(cost=ns_val - ms_val)
        return Value(cost=1.0)

    def _get_applicable_actions_from(
        self, memory: D.T_memory[D.T_state]
    ) -> D.T_agent[Space[D.T_event]]:
        cpp_actions = self._aops_gen.get_applicable_actions(memory.to_cpp())
        return ListSpace([PDDLAction(a, self._task) for a in cpp_actions])

    def _get_goals_(self) -> D.T_agent[Space[D.T_observation]]:
        return ImplicitSpace(lambda s: self._goal_checker.is_goal(s.to_cpp()))

    def _is_terminal(self, state: D.T_state) -> D.T_agent[D.T_predicate]:
        if self._goal_checker.is_goal(state.to_cpp()):
            return True
        actions = self._aops_gen.get_applicable_actions(state.to_cpp())
        return len(actions) == 0

    def _get_action_space_(self) -> D.T_agent[Space[D.T_event]]:
        return ImplicitSpace(lambda a: isinstance(a, PDDLAction))

    def _get_observation_space_(self) -> D.T_agent[Space[D.T_observation]]:
        return ImplicitSpace(lambda s: isinstance(s, PDDLState))

    @property
    def task(self):
        return self._task

    @property
    def cpp_domain(self):
        return self._reader.domains[0]

    @property
    def cpp_problem(self):
        return self._reader.problems[0]


class TPDDLState:
    """Immutable, hashable PDDL+ temporal state."""

    def __init__(self, cpp_state, cost_function_ids=frozenset()):
        self._cpp_state = cpp_state
        self._atoms = tuple(
            frozenset(tuple(t) for t in s) for s in cpp_state.get_atoms()
        )
        fluents_raw = cpp_state.get_fluents()
        self._fluents = tuple(
            frozenset((tuple(k), v) for k, v in f.items())
            if i not in cost_function_ids
            else frozenset()
            for i, f in enumerate(fluents_raw)
        )
        self._time = cpp_state.time
        ada_list = cpp_state.get_active_durative_actions()
        self._active_da = tuple(
            (d["action_id"], d["start_time"], d["end_time"]) for d in ada_list
        )
        self._hash = hash((self._atoms, self._fluents, self._time, self._active_da))

    def to_cpp(self):
        return self._cpp_state

    @property
    def time(self):
        return self._time

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return (
            isinstance(other, TPDDLState)
            and self._atoms == other._atoms
            and self._fluents == other._fluents
            and self._time == other._time
            and self._active_da == other._active_da
        )

    def __repr__(self):
        return f"TPDDLState(time={self._time:.4f}, atoms={self._atoms})"


class TPDDLAction:
    """Temporal PDDL action: instantaneous, durative-start, or noop."""

    NOOP = 0
    INSTANTANEOUS = 1
    DURATIVE_START = 2

    def __init__(self, kind, cpp_ground_action=None, task=None):
        self.kind = kind
        self._cpp_action = cpp_ground_action
        self._task = task
        if cpp_ground_action is not None:
            self.action_id = cpp_ground_action.action_id
            self.arguments = tuple(cpp_ground_action.arguments)
        else:
            self.action_id = -1
            self.arguments = ()

    def to_cpp(self):
        return self._cpp_action

    def __hash__(self):
        return hash((self.kind, self.action_id, self.arguments))

    def __eq__(self, other):
        return (
            isinstance(other, TPDDLAction)
            and self.kind == other.kind
            and self.action_id == other.action_id
            and self.arguments == other.arguments
        )

    def __repr__(self):
        if self.kind == self.NOOP:
            return "(noop)"
        label = "act" if self.kind == self.INSTANTANEOUS else "start-da"
        if self._task is not None and self._cpp_action is not None:
            if self.kind == self.INSTANTANEOUS:
                name = self._task.action_name(self.action_id)
            else:
                name = self._task.durative_action_name(self.action_id)
            args = " ".join(self._task.object_name(a) for a in self.arguments)
            if args:
                return f"({label}:{name} {args})"
            return f"({label}:{name})"
        return f"({label}:{self.action_id})"


class TPDDLDomain(DeterministicPlanningDomain, UnrestrictedActions):
    """PDDL+ temporal planning domain with processes, events, and continuous time.

    # Parameters
    domain_path: Path to the PDDL+ domain file.
    problem_path: Path to the PDDL+ problem file.
    mode: Transition mode — "time_stepping" or "event_driven".
    dt: Time step size for time_stepping mode.
    max_event_lookahead: Maximum time to look ahead for events.
    epsilon: Numerical precision for event detection.
    max_cascade_iterations: Maximum event cascade iterations.
    """

    T_state = TPDDLState
    T_observation = T_state
    T_event = TPDDLAction

    def __init__(
        self,
        domain_path: str,
        problem_path: str,
        mode: str = "time_stepping",
        dt: float = 0.1,
        max_event_lookahead: float = 1e6,
        epsilon: float = 1e-9,
        max_cascade_iterations: int = 100,
        use_z3: bool = True,
    ):
        self._mode = mode
        self._dt = dt
        self._reader = PDDLReader(domain_path, problem_path)
        self._task = CppTask(self._reader.domains[0], self._reader.problems[0])

        event_time_finder = None
        if mode == "event_driven" and use_z3 and _HAS_Z3_PYTHON:
            self._z3_finder = Z3EventTimeFinder(
                self._task, domain_path, max_event_lookahead, epsilon
            )
            event_time_finder = self._z3_finder.find_next_event_time

        self._sim = CppTemporalSimulator(
            self._task,
            epsilon=epsilon,
            max_event_lookahead=max_event_lookahead,
            max_cascade_iterations=max_cascade_iterations,
            event_time_finder=event_time_finder,
        )

        if hasattr(self, "_z3_finder"):
            self._z3_finder.set_sim(self._sim)

        self._cost_function_ids = set()
        tc = self._task.total_cost_function()
        if tc >= 0:
            self._cost_function_ids.add(tc)
            self._total_cost_idx = tc
        else:
            self._total_cost_idx = -1
        rw = self._task.reward_function()
        if rw >= 0:
            self._cost_function_ids.add(rw)
        self._cost_function_ids = frozenset(self._cost_function_ids)

    def _get_initial_state_(self) -> D.T_state:
        return TPDDLState(self._task.initial_state(), self._cost_function_ids)

    def _get_next_state(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
    ) -> D.T_state:
        cpp_state = memory.to_cpp()

        if action.kind == TPDDLAction.NOOP:
            if self._mode == "time_stepping":
                cpp_next = self._sim.time_step(cpp_state, self._dt)
            else:
                cpp_next = self._sim.event_step(cpp_state)
        elif action.kind == TPDDLAction.INSTANTANEOUS:
            if self._mode == "time_stepping":
                cpp_next = self._sim.time_step(cpp_state, self._dt, action.to_cpp())
            else:
                cpp_next = self._sim.event_step(cpp_state, action.to_cpp())
        elif action.kind == TPDDLAction.DURATIVE_START:
            cpp_next = self._sim.start_durative_action(cpp_state, action.to_cpp())
            if self._mode == "time_stepping":
                cpp_next = self._sim.time_step(cpp_next, self._dt)
            else:
                cpp_next = self._sim.event_step(cpp_next)
        else:
            cpp_next = cpp_state

        return TPDDLState(cpp_next, self._cost_function_ids)

    def _get_transition_value(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
        next_state: Optional[D.T_state] = None,
    ) -> D.T_agent[Value[D.T_value]]:
        if next_state is not None:
            dt = next_state.time - memory.time
            return Value(cost=max(dt, 0.0))
        return Value(cost=self._dt)

    def _get_applicable_actions_from(
        self, memory: D.T_memory[D.T_state]
    ) -> D.T_agent[Space[D.T_event]]:
        cpp_state = memory.to_cpp()
        actions = []

        actions.append(TPDDLAction(TPDDLAction.NOOP))

        for ga in self._sim.get_applicable_actions(cpp_state):
            actions.append(TPDDLAction(TPDDLAction.INSTANTANEOUS, ga, self._task))

        for ga in self._sim.get_applicable_durative_actions(cpp_state):
            actions.append(TPDDLAction(TPDDLAction.DURATIVE_START, ga, self._task))

        return ListSpace(actions)

    def _get_goals_(self) -> D.T_agent[Space[D.T_observation]]:
        return ImplicitSpace(lambda s: self._sim.is_goal(s.to_cpp()))

    def _is_terminal(self, state: D.T_state) -> D.T_agent[D.T_predicate]:
        if self._sim.is_goal(state.to_cpp()):
            return True
        actions = self._get_applicable_actions_from(state)
        has_non_noop = any(a.kind != TPDDLAction.NOOP for a in actions.get_elements())
        return not has_non_noop

    def _get_action_space_(self) -> D.T_agent[Space[D.T_event]]:
        return ImplicitSpace(lambda a: isinstance(a, TPDDLAction))

    def _get_observation_space_(self) -> D.T_agent[Space[D.T_observation]]:
        return ImplicitSpace(lambda s: isinstance(s, TPDDLState))

    @property
    def task(self):
        return self._task

    @property
    def simulator(self):
        return self._sim

    @property
    def cpp_domain(self):
        return self._reader.domains[0]

    @property
    def cpp_problem(self):
        return self._reader.problems[0]


class Z3EventTimeFinder:
    """Uses Z3 SMT solver to analytically compute event trigger times.

    Parses event numeric preconditions from the PDDL domain file, estimates
    fluent change rates from active processes via numerical differentiation,
    and uses Z3's exact rational arithmetic to find the minimum time at which
    any event precondition is satisfied under a linear trajectory model.

    Falls back to binary search when Z3 constraints cannot be built (e.g.
    non-linear dynamics) or when the Z3 solution fails verification.
    """

    def __init__(self, task, domain_path, max_event_lookahead, epsilon):
        import z3 as _z3_mod

        self._z3 = _z3_mod
        self._task = task
        self._sim = None
        self._max_lookahead = max_event_lookahead
        self._epsilon = epsilon
        self._event_numeric_conditions = self._parse_event_numeric_conditions(
            domain_path
        )

    def set_sim(self, sim):
        self._sim = sim

    @staticmethod
    def _tokenize(text):
        tokens = []
        i = 0
        while i < len(text):
            c = text[i]
            if c in "()":
                tokens.append(c)
                i += 1
            elif c == ";":
                while i < len(text) and text[i] != "\n":
                    i += 1
            elif c.isspace():
                i += 1
            else:
                j = i
                while j < len(text) and text[j] not in "() \t\n\r;":
                    j += 1
                tokens.append(text[i:j])
                i = j
        return tokens

    @staticmethod
    def _parse_sexpr(tokens, pos):
        if pos >= len(tokens):
            return None, pos
        if tokens[pos] == "(":
            result = []
            pos += 1
            while pos < len(tokens) and tokens[pos] != ")":
                child, pos = Z3EventTimeFinder._parse_sexpr(tokens, pos)
                if child is not None:
                    result.append(child)
            if pos < len(tokens):
                pos += 1
            return result, pos
        return tokens[pos], pos + 1

    def _parse_numeric_expr(self, expr):
        """Parse into ('const', value) or ('func', name, param_names)."""
        if isinstance(expr, str):
            try:
                return ("const", float(expr))
            except ValueError:
                if not expr.startswith("?") and not expr.startswith(":"):
                    return ("func", expr, [])
                return None
        if isinstance(expr, list) and expr:
            func_name = expr[0]
            params = [p for p in expr[1:] if isinstance(p, str)]
            return ("func", func_name, params)
        return None

    def _extract_numeric_comparisons(self, formula):
        if not isinstance(formula, list) or not formula:
            return []
        op = formula[0]
        if op == "and":
            result = []
            for child in formula[1:]:
                result.extend(self._extract_numeric_comparisons(child))
            return result
        if op in (">=", "<=", ">", "<", "=") and len(formula) == 3:
            lhs = self._parse_numeric_expr(formula[1])
            rhs = self._parse_numeric_expr(formula[2])
            if lhs is not None and rhs is not None:
                return [(op, lhs, rhs)]
        return []

    def _parse_event_numeric_conditions(self, domain_path):
        with open(domain_path) as f:
            content = f.read()
        tokens = self._tokenize(content)
        domain_expr, _ = self._parse_sexpr(tokens, 0)
        if not isinstance(domain_expr, list):
            return {}
        events = {}
        for item in domain_expr:
            if not isinstance(item, list) or not item or item[0] != ":event":
                continue
            event_name = item[1] if len(item) > 1 else None
            if event_name is None:
                continue
            precondition = None
            i = 2
            while i < len(item):
                if item[i] == ":precondition" and i + 1 < len(item):
                    precondition = item[i + 1]
                    i += 2
                else:
                    i += 1
            if precondition is not None:
                conditions = self._extract_numeric_comparisons(precondition)
                if conditions:
                    events[event_name] = conditions
        return events

    def _estimate_rates(self, cpp_state):
        tiny_dt = min(1e-3, self._max_lookahead * 1e-9)
        s2 = self._sim.integrate_processes(cpp_state, tiny_dt)
        f0 = cpp_state.get_fluents()
        f1 = s2.get_fluents()
        rates = {}
        for fid in range(len(f0)):
            for key in f0[fid]:
                v0 = f0[fid][key]
                v1 = f1[fid].get(key, v0)
                if abs(v1 - v0) > 1e-15:
                    rates[(fid, key)] = (v1 - v0) / tiny_dt
        return rates

    def _build_z3_trajectory(self, expr_desc, f0, rates, t_var):
        """Build Z3 expression(s) for a numeric expression descriptor.

        Returns either a single Z3 expression (for no-arg functions or
        constants) or a dict {arg_tuple: z3_expr} for parameterized functions.
        """
        z3 = self._z3
        kind = expr_desc[0]
        if kind == "const":
            return z3.RealVal(str(expr_desc[1]))
        func_name = expr_desc[1]
        try:
            func_id = self._task.function_id(func_name)
        except Exception:
            return None
        if not f0[func_id]:
            return None
        results = {}
        for key, val in f0[func_id].items():
            rate = rates.get((func_id, key), 0.0)
            results[key] = z3.RealVal(str(val)) + z3.RealVal(str(rate)) * t_var
        if len(results) == 1 and () in results:
            return results[()]
        return results

    def _solve_one_constraint(self, t_var, comp_op, lhs_z3, rhs_z3):
        z3 = self._z3
        ops = {
            ">=": lambda l, r: l >= r,
            "<=": lambda l, r: l <= r,
            ">": lambda l, r: l > r,
            "<": lambda l, r: l < r,
            "=": lambda l, r: l == r,
        }
        if comp_op not in ops:
            return None
        opt = z3.Optimize()
        opt.add(t_var > 0)
        opt.add(t_var <= z3.RealVal(str(self._max_lookahead)))
        opt.add(ops[comp_op](lhs_z3, rhs_z3))
        opt.minimize(t_var)
        if opt.check() == z3.sat:
            t_val = opt.model()[t_var]
            if t_val is not None:
                return float(t_val.as_fraction())
        return None

    def find_next_event_time(self, cpp_state):
        if self._sim is None:
            return self._max_lookahead
        triggered = self._sim.get_triggered_events(cpp_state)
        if triggered:
            return self._epsilon
        active_procs = self._sim.get_active_processes(cpp_state)
        if not active_procs:
            return self._max_lookahead
        rates = self._estimate_rates(cpp_state)
        if not rates:
            return self._max_lookahead

        f0 = cpp_state.get_fluents()
        z3 = self._z3
        t = z3.Real("t")
        best_time = self._max_lookahead

        for _event_name, conditions in self._event_numeric_conditions.items():
            for comp_op, lhs_desc, rhs_desc in conditions:
                lhs = self._build_z3_trajectory(lhs_desc, f0, rates, t)
                rhs = self._build_z3_trajectory(rhs_desc, f0, rates, t)
                if lhs is None or rhs is None:
                    continue
                pairs = []
                if isinstance(lhs, dict) and isinstance(rhs, dict):
                    for k in lhs:
                        if k in rhs:
                            pairs.append((lhs[k], rhs[k]))
                elif isinstance(lhs, dict):
                    for lz in lhs.values():
                        pairs.append((lz, rhs))
                elif isinstance(rhs, dict):
                    for rz in rhs.values():
                        pairs.append((lhs, rz))
                else:
                    pairs.append((lhs, rhs))
                for lz, rz in pairs:
                    tv = self._solve_one_constraint(t, comp_op, lz, rz)
                    if tv is not None and tv < best_time:
                        best_time = tv

        if best_time < self._max_lookahead:
            s_check = self._sim.integrate_processes(cpp_state, best_time)
            if self._sim.get_triggered_events(s_check):
                return max(best_time, self._epsilon)

        return self._fallback_binary_search(cpp_state)

    def _fallback_binary_search(self, cpp_state):
        s_far = self._sim.integrate_processes(cpp_state, self._max_lookahead)
        if not self._sim.get_triggered_events(s_far):
            return self._max_lookahead
        lo, hi = 0.0, self._max_lookahead
        while hi - lo > self._epsilon:
            mid = (lo + hi) / 2.0
            s_mid = self._sim.integrate_processes(cpp_state, mid)
            if self._sim.get_triggered_events(s_mid):
                hi = mid
            else:
                lo = mid
        return hi
