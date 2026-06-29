# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

from skdecide import Domain, Solver
from skdecide.builders.domain import (
    Actions,
    FullyObservable,
    Goals,
    Markovian,
    PositiveCosts,
    Sequential,
    Simulation,
    SingleAgent,
)
from skdecide.builders.solver import DeterministicPolicies, FromAnyState, Utilities

try:
    from skdecide.hub.__skdecide_hub_cpp import (
        _PPDDLDetHindsightSolver_ as CppPPDDLDetHindsightSolver,
    )
    from skdecide.hub.domain.pddl.domain import PDDLAction

    class D(
        Domain,
        SingleAgent,
        Sequential,
        Simulation,
        Actions,
        Goals,
        Markovian,
        FullyObservable,
        PositiveCosts,
    ):
        pass

    class PPDDLDetHindsight(Solver, DeterministicPolicies, Utilities, FromAnyState):
        """Hindsight optimization for probabilistic PDDL (PPDDL) domains
        with a pluggable inner deterministic solver.

        At each state, enumerates applicable actions, samples W random
        determinizations of the PDDL effect tree, solves each
        deterministic problem from each action's successor, averages
        Q-values across scenarios, and returns the action with the
        lowest average cost.

        Counterpart of FFReplan, but uses hindsight optimization
        instead of replanning, and supports pluggable inner solvers.

        Requires a PPDDLDomain.

        # Reference
        Yoon, S. W., Fern, A., & Givan, R. (2008). Probabilistic Planning
        via Determinization in Hindsight. In *Proc. AAAI*, pp. 1010-1016.
        """

        T_domain = D

        def __init__(
            self,
            domain_factory: Callable[[], Domain],
            inner_solver_factory: Optional[Callable[[], tuple[str, dict]]] = None,
            sample_width: int = 30,
            dead_end_cost: float = 1e9,
            max_steps: int = 10000,
            discount: float = 0.99,
            epsilon: float = 1e-3,
            parallel: bool = False,
            callback: Callable[["PPDDLDetHindsight"], bool] = lambda slv: False,
            verbose: bool = False,
        ) -> None:
            """Construct a PPDDLDetHindsight solver instance.

            # Parameters
            domain_factory: Lambda to create a PPDDL domain instance.
            inner_solver_factory: Factory returning (solver_name, params_dict).
                Defaults to ("FF", {}). Use get_available_pddl_inner_solvers()
                to list registered solvers.
            sample_width: Number of random determinization scenarios to
                sample at each step. Defaults to 30.
            dead_end_cost: Cost penalty for dead-end states where the
                inner solver fails. Defaults to 1e9.
            max_steps: Maximum total simulation steps. Defaults to 10000.
            discount: Discount factor for value evaluation convergence.
                Required because reachable dead-end states with cost
                dead_end_cost need contraction. Defaults to 0.99.
            epsilon: Convergence threshold for value evaluation residual.
                Defaults to 1e-3.
            parallel: Parallelize scenario evaluation. Defaults to False.
            callback: Called after each hindsight evaluation; return True
                to stop. Defaults to never stop.
            verbose: Log progress messages. Defaults to False.
            """
            Solver.__init__(self, domain_factory=domain_factory)
            inner_solver_name, inner_solver_params = (
                inner_solver_factory()
                if inner_solver_factory is not None
                else ("FF", {})
            )
            self._inner_solver_name = inner_solver_name
            self._inner_solver_params = inner_solver_params
            self._sample_width = sample_width
            self._dead_end_cost = dead_end_cost
            self._max_steps = max_steps
            self._discount = discount
            self._epsilon = epsilon
            self._parallel = parallel
            self._callback = callback
            self._verbose = verbose

        def _solve_from(self, memory: D.T_memory[D.T_state]) -> None:
            """Run the PPDDL-DetHindsight algorithm from a given PDDL state.

            # Parameters
            memory: PDDL state from which to start hindsight optimization.
            """
            if not hasattr(self, "_cpp_solver"):
                domain = self._domain_factory()
                if not hasattr(domain, "_task"):
                    raise TypeError(
                        "PPDDLDetHindsight solver requires a PPDDLDomain "
                        "(with _task attribute)"
                    )
                self._task = domain._task
                self._cpp_solver = CppPPDDLDetHindsightSolver(
                    self,
                    self._task,
                    self._inner_solver_name,
                    self._parallel,
                    self._sample_width,
                    self._dead_end_cost,
                    self._max_steps,
                    self._discount,
                    self._epsilon,
                    self._callback,
                    self._verbose,
                    self._inner_solver_params,
                )
            self._cpp_solver.solve(memory.to_cpp())

        def _is_solution_defined_for(
            self, observation: D.T_agent[D.T_observation]
        ) -> bool:
            return self._cpp_solver.is_solution_defined_for(observation.to_cpp())

        def _get_next_action(
            self,
            observation: D.T_agent[D.T_observation],
            domain: Optional[Domain] = None,
        ) -> D.T_agent[D.T_concurrency[D.T_event]]:
            action = self._cpp_solver.get_next_action(observation.to_cpp())
            if action is None:
                return None
            return PDDLAction(action, self._task)

        def _get_utility(self, observation: D.T_agent[D.T_observation]) -> D.T_value:
            return self._cpp_solver.get_best_value(observation.to_cpp())

        def get_nb_steps(self) -> int:
            """Get the total number of simulation steps taken."""
            return self._cpp_solver.get_nb_steps()

        def get_solving_time(self) -> int:
            """Get the total solving time in milliseconds."""
            return self._cpp_solver.get_solving_time()

        def get_explored_states(self) -> set:
            """Get the set of states explored during the last hindsight evaluation."""
            return self._cpp_solver.get_explored_states()

        def get_terminal_states(self) -> set:
            """Get the set of terminal states (goals or dead ends) from the last hindsight evaluation."""
            return self._cpp_solver.get_terminal_states()

except ImportError:
    print(
        "Scikit-decide C++ hub library not found. Please check it is "
        'installed in "skdecide/hub".'
    )
    raise
