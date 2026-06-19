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
    SingleAgent,
    UncertainTransitions,
)
from skdecide.builders.solver import (
    DeterministicPolicies,
    FromAnyState,
    ParallelSolver,
    Utilities,
)
from skdecide.core import Value

try:
    from skdecide.hub.__skdecide_hub_cpp import (
        _SSPDetHindsightSolver_ as sspdethindsight_solver,
    )

    class D(
        Domain,
        SingleAgent,
        Sequential,
        UncertainTransitions,
        Actions,
        Goals,
        Markovian,
        FullyObservable,
        PositiveCosts,
    ):
        pass

    class SSPDetHindsight(
        ParallelSolver, Solver, DeterministicPolicies, Utilities, FromAnyState
    ):
        """Hindsight optimization for stochastic shortest path (SSP) domains.

        Implements the Hindsight Optimization algorithm from Yoon, Fern &
        Givan (AAAI 2008). At each state, enumerates applicable actions,
        samples W random determinizations of the stochastic domain, solves
        each deterministic problem from each action's successor, averages
        Q-values across scenarios, and returns the action with the lowest
        average cost.

        This is an online policy: hindsight evaluation happens at every
        ``get_best_action()`` call.

        Uses transition-level determinization: for each scenario, each
        call to ``get_next_state(s, a)`` samples one outcome from the
        probability distribution over successors.

        # Reference
        Yoon, S. W., Fern, A., & Givan, R. (2008). Probabilistic Planning
        via Determinization in Hindsight. In *Proc. AAAI*, pp. 1010-1016.
        """

        T_domain = D

        def __init__(
            self,
            domain_factory: Callable[[], D],
            heuristic: Callable[[D, D.T_state], D.T_agent[Value[D.T_value]]] = lambda d,
            s: Value(cost=0),
            inner_solver_factory: Optional[Callable[[], tuple[str, dict]]] = None,
            sample_width: int = 30,
            dead_end_cost: float = 1000.0,
            max_steps: int = 10000,
            parallel: bool = False,
            shared_memory_proxy=None,
            callback: Callable[[SSPDetHindsight, Optional[int]], bool] = lambda slv,
            i=None: False,
            verbose: bool = False,
        ) -> None:
            """Construct an SSPDetHindsight solver instance.

            # Parameters
            domain_factory: Lambda to create a domain instance.
            heuristic: Function h(domain, state) -> Value estimating
                cost-to-go. Defaults to Value(cost=0).
            inner_solver_factory: Callable returning a (name, params) tuple
                specifying the inner deterministic solver. Available inner
                solvers: "Astar", "EHC". Defaults to ``lambda: ("Astar", {})``.
            sample_width: Number of random determinization scenarios to
                sample at each step. Defaults to 30.
            dead_end_cost: Cost penalty assigned when the inner solver
                cannot find a plan from a successor state. Defaults to
                1000.0.
            max_steps: Maximum total simulation steps. Defaults to 10000.
            parallel: Parallelize domain calls. Defaults to False.
            shared_memory_proxy: Optional shared memory proxy.
            callback: Called after each hindsight evaluation; return True
                to stop. Defaults to never stop.
            verbose: Log progress messages. Defaults to False.
            """
            if inner_solver_factory is None:
                inner_solver_factory = lambda: ("Astar", {})
            inner_solver, inner_solver_params = inner_solver_factory()

            Solver.__init__(self, domain_factory=domain_factory)
            ParallelSolver.__init__(
                self,
                parallel=parallel,
                shared_memory_proxy=shared_memory_proxy,
            )
            self._lambdas = [heuristic]
            self._ipc_notify = True

            self._solver = sspdethindsight_solver(
                solver=self,
                domain=self.get_domain(),
                goal_checker=lambda d, s: d.is_goal(s),
                heuristic=lambda d, s: heuristic(d, s),
                inner_solver=inner_solver,
                inner_solver_params=inner_solver_params,
                sample_width=sample_width,
                dead_end_cost=dead_end_cost,
                max_steps=max_steps,
                parallel=parallel,
                callback=callback,
                verbose=verbose,
            )

        def close(self):
            """Joins parallel domain processes."""
            if self._parallel:
                self._solver.close()
            ParallelSolver.close(self)

        def _solve_from(self, memory: D.T_memory[D.T_state]) -> None:
            """Run hindsight optimization from a given state.

            # Parameters
            memory: State from which to start.
            """
            self._solver.solve(memory)

        def _is_solution_defined_for(
            self, observation: D.T_agent[D.T_observation]
        ) -> bool:
            return self._solver.is_solution_defined_for(observation)

        def _get_next_action(
            self,
            observation: D.T_agent[D.T_observation],
            domain: Optional[Domain] = None,
        ) -> D.T_agent[D.T_concurrency[D.T_event]]:
            """Get the best action for a given state via hindsight optimization.

            !!! warning
                Returns a random action if no action is defined in the given
                state.

            # Parameters
            observation: State for which the best action is requested.

            # Returns
            Best action from hindsight evaluation.
            """
            action = self._solver.get_next_action(observation)
            if action is None:
                print(
                    "\x1b[3;33;40m"
                    + "No best action found in observation "
                    + str(observation)
                    + ", applying random action"
                    + "\x1b[0m"
                )
                return self.call_domain_method("get_action_space").sample()
            return action

        def _get_utility(self, observation: D.T_agent[D.T_observation]) -> D.T_value:
            return self._solver.get_utility(observation)

        def get_nb_steps(self) -> int:
            """Get the total number of simulation steps taken."""
            return self._solver.get_nb_steps()

        def get_solving_time(self) -> int:
            """Get the total solving time in milliseconds."""
            return self._solver.get_solving_time()

        def get_explored_states(self) -> set:
            """Get the set of states explored during the last hindsight evaluation."""
            return self._solver.get_explored_states()

        def get_terminal_states(self) -> set:
            """Get the set of terminal states (goals or dead ends) from the last hindsight evaluation."""
            return self._solver.get_terminal_states()

except ImportError:
    print(
        "Scikit-decide C++ hub library not found. Please check it is "
        'installed in "skdecide/hub".'
    )
    raise
