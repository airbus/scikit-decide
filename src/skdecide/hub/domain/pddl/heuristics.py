# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from skdecide import Value
from skdecide.hub.__skdecide_hub_cpp import (
    _PDDL_DeleteRelaxationHeuristic_ as CppDeleteRelaxation,
)
from skdecide.hub.__skdecide_hub_cpp import _PDDL_FFHeuristic_ as CppFFHeuristic
from skdecide.hub.__skdecide_hub_cpp import _PDDL_HeuristicMode_ as CppMode


def atom_to_str(atom, task):
    """Format a FlatAtomKey as a human-readable string using task metadata.

    Args:
        atom: A _PDDL_FlatAtomKey_ object with predicate_id and args.
        task: The PDDL Task providing predicate_name() and object_name().

    Returns:
        A string like "on(block1, block2)".
    """
    name = task.predicate_name(atom.predicate_id)
    args = ", ".join(task.object_name(a) for a in atom.args)
    return f"{name}({args})" if args else name


class HMax:
    """h_max delete-relaxation heuristic (admissible).

    Takes the MAX over goal atom costs. Lower bound on optimal plan cost.
    Pre-grounds all reachable actions at construction time via clingo,
    then computes heuristic values via fast array-based forward chaining.

    For deterministic domains (PDDL): computes h_max as defined in
    "Planning as Heuristic Search" (Bonet & Geffner, Artificial
    Intelligence, 2001).

    For probabilistic domains (PPDDL):
    - discount_factor=1.0 (SSP): applies Proposition 1 from
      "Extending Classical Planning Heuristics to Probabilistic Planning
      with Dead-Ends" (Teichteil-Königsbuch, Vidal & Infantes, AAAI 2011)
      — h⁺_max on the implicit all-outcome determinization.
    - discount_factor<1.0 (DSSP): applies Theorem 2 — discounted formula
      h^γ_max(s) = c_m(s) · (1 − γ^{h^{1,+}_max(s)}) / (1 − γ).
    """

    def __init__(self, task, discount_factor=1.0, dead_end_cost=1e9, verbose=False):
        if discount_factor < 1.0:
            reqs = task.domain().get_requirements()
            if not reqs.has_probabilistic_effects():
                raise ValueError(
                    "discount_factor < 1.0 requires a domain with "
                    "probabilistic effects (:probabilistic-effects)"
                )
        self._cpp = CppDeleteRelaxation(
            task, CppMode.HMAX, discount_factor, dead_end_cost, verbose
        )

    def __call__(self, state=None):
        """Compute h_max for a state, or return a solver-compatible callable.

        - hmax(state): returns the heuristic value (float) for the given state.
        - hmax(): returns a callable ``(domain, state) -> Value`` suitable for
          passing as ``heuristic=hmax()`` to scikit-decide solvers like Astar.
        """
        if state is None:
            cpp = self._cpp
            return lambda d, s: Value(cost=cpp.compute(s.to_cpp()))
        return self._cpp.compute(state)

    def compute_detailed(self, state):
        """Return detailed heuristic information for the given state.

        Returns a dict with:
        - heuristic_value: the h_max value
        - atom_costs: list of (FlatAtomKey, cost) for reachable atoms
        - goal_atom_costs: list of (FlatAtomKey, cost) for goal atoms
        """
        return self._cpp.compute_detailed(state)

    @property
    def num_atoms(self):
        return self._cpp.num_atoms()

    @property
    def num_relaxed_actions(self):
        return self._cpp.num_relaxed_actions()


class HAdd:
    """h_add delete-relaxation heuristic (non-admissible, informative).

    SUM over goal atom costs. Assumes subgoals are independent.
    Pre-grounds all reachable actions at construction time via clingo,
    then computes heuristic values via fast array-based forward chaining.

    For deterministic domains (PDDL): computes h_add as defined in
    "Planning as Heuristic Search" (Bonet & Geffner, Artificial
    Intelligence, 2001).

    For probabilistic domains (PPDDL):
    - discount_factor=1.0 (SSP): applies Proposition 1 from
      "Extending Classical Planning Heuristics to Probabilistic Planning
      with Dead-Ends" (Teichteil-Königsbuch, Vidal & Infantes, AAAI 2011)
      — h⁺_add on the implicit all-outcome determinization.
    - discount_factor<1.0 (DSSP): applies Theorem 2 — discounted formula
      h^γ_add(s) = c_m(s) · (1 − γ^{h^{1,+}_add(s)}) / (1 − γ).
    """

    def __init__(self, task, discount_factor=1.0, dead_end_cost=1e9, verbose=False):
        if discount_factor < 1.0:
            reqs = task.domain().get_requirements()
            if not reqs.has_probabilistic_effects():
                raise ValueError(
                    "discount_factor < 1.0 requires a domain with "
                    "probabilistic effects (:probabilistic-effects)"
                )
        self._cpp = CppDeleteRelaxation(
            task, CppMode.HADD, discount_factor, dead_end_cost, verbose
        )

    def __call__(self, state=None):
        """Compute h_add for a state, or return a solver-compatible callable.

        - hadd(state): returns the heuristic value (float) for the given state.
        - hadd(): returns a callable ``(domain, state) -> Value`` suitable for
          passing as ``heuristic=hadd()`` to scikit-decide solvers like Astar.
        """
        if state is None:
            cpp = self._cpp
            return lambda d, s: Value(cost=cpp.compute(s.to_cpp()))
        return self._cpp.compute(state)

    def compute_detailed(self, state):
        """Return detailed heuristic information for the given state.

        Returns a dict with:
        - heuristic_value: the h_add value
        - atom_costs: list of (FlatAtomKey, cost) for reachable atoms
        - goal_atom_costs: list of (FlatAtomKey, cost) for goal atoms
        """
        return self._cpp.compute_detailed(state)

    @property
    def num_atoms(self):
        return self._cpp.num_atoms()

    @property
    def num_relaxed_actions(self):
        return self._cpp.num_relaxed_actions()


class HFF:
    """h_FF delete-relaxation heuristic.

    Builds a relaxed planning graph via h_add forward chaining, then
    extracts a relaxed plan backwards from the goal as described in:

        Hoffmann, J. and Nebel, B. (2001). The FF Planning System:
        Fast Plan Generation Through Heuristic Search.
        Journal of Artificial Intelligence Research, 14, 253-302.

    h_FF = total cost of unique actions in the relaxed plan.
    Also identifies helpful actions (relaxed-plan actions whose
    preconditions are all satisfied in the current state).

    For probabilistic domains (PPDDL):
    - discount_factor=1.0 (SSP): h⁺_FF on all-outcome determinization.
    - discount_factor<1.0 (DSSP): h^γ_FF via Theorem 2 from
      "Extending Classical Planning Heuristics to Probabilistic Planning
      with Dead-Ends" (Teichteil-Königsbuch, Vidal & Infantes, AAAI 2011).
    """

    def __init__(self, task, discount_factor=1.0, dead_end_cost=1e9, verbose=False):
        if discount_factor < 1.0:
            reqs = task.domain().get_requirements()
            if not reqs.has_probabilistic_effects():
                raise ValueError(
                    "discount_factor < 1.0 requires a domain with "
                    "probabilistic effects (:probabilistic-effects)"
                )
        self._cpp = CppFFHeuristic(task, discount_factor, dead_end_cost, verbose)

    def __call__(self, state=None):
        """Compute h_FF for a state, or return a solver-compatible callable.

        - hff(state): returns the heuristic value (float) for the given state.
        - hff(): returns a callable ``(domain, state) -> Value`` suitable for
          passing as ``heuristic=hff()`` to scikit-decide solvers like Astar.
        """
        if state is None:
            cpp = self._cpp
            return lambda d, s: Value(cost=cpp.compute(s.to_cpp()))
        return self._cpp.compute(state)

    def compute_with_helpful(self, state):
        """Return (h_value, [GroundAction]) for the given state."""
        return self._cpp.compute_with_helpful(state)

    def helpful_actions(self, state):
        """Return helpful actions (list of GroundAction) for the given state."""
        return self._cpp.compute_with_helpful(state)[1]

    def compute_detailed(self, state):
        """Return detailed heuristic information for the given state.

        Returns a dict with:
        - heuristic_value: the h_FF value
        - atom_costs: list of (FlatAtomKey, cost) for reachable atoms
        - goal_atom_costs: list of (FlatAtomKey, cost) for goal atoms
        - relaxed_plan_actions: list of GroundAction in the relaxed plan
        - helpful_actions: list of GroundAction (relaxed-plan actions
          whose preconditions are all satisfied in the current state)
        - marked_atoms: list of FlatAtomKey extracted during backward pass
        """
        return self._cpp.compute_detailed(state)

    @property
    def num_atoms(self):
        return self._cpp.num_atoms()

    @property
    def num_relaxed_actions(self):
        return self._cpp.num_relaxed_actions()
