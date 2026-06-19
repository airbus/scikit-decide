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
        _SSPPlanMergerSolver_ as sspplanmerger_solver,
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

    class SSPPlanMerger(
        ParallelSolver, Solver, DeterministicPolicies, Utilities, FromAnyState
    ):
        """SSP plan-merging solver: iteratively determinizes a stochastic
        domain, plans from terminal states, and merges plans into a policy
        until Monte-Carlo assessment shows the replanning probability is
        below a threshold.

        Supports three determinization strategies:
        - ``"most_probable_outcome"``: picks the most likely successor
        - ``"all_outcomes"``: creates a deterministic action per outcome
        - ``"random_outcome"``: picks a random successor

        Optional discounted value iteration on the policy graph treats
        terminal states as absorbing dead-ends and optimizes actions.

        # Reference
        Teichteil-Königsbuch, F., Kuter, U., & Infantes, G. (2010).
        RFF: A Robust, FF-Based MDP Planning Algorithm for Generating
        Policies with Low Probability of Failure. In *Proc. AAMAS*.
        """

        T_domain = D

        def __init__(
            self,
            domain_factory: Callable[[], D],
            heuristic: Callable[[D, D.T_state], D.T_agent[Value[D.T_value]]] = lambda d,
            s: Value(cost=0),
            determinization: str = "most_probable_outcome",
            inner_solver_factory: Optional[Callable[[], tuple[str, dict]]] = None,
            rho: float = 0.1,
            mc_samples: int = 100,
            max_iterations: int = 50,
            max_steps: int = 10000,
            dead_end_cost: float = 1e9,
            optimize_policy_graph: bool = False,
            discount: float = 0.99,
            epsilon: float = 1e-3,
            continuous_planning: bool = False,
            parallel: bool = False,
            shared_memory_proxy=None,
            callback: Callable[[SSPPlanMerger, Optional[int]], bool] = lambda slv,
            i=None: False,
            verbose: bool = False,
        ) -> None:
            """Construct an SSPPlanMerger solver instance.

            # Parameters
            domain_factory: Lambda to create a domain instance.
            heuristic: Function h(domain, state) -> Value estimating cost-to-go.
                Defaults to Value(cost=0).
            determinization: Determinization strategy. One of
                ``"most_probable_outcome"``, ``"all_outcomes"``,
                ``"random_outcome"``. Defaults to ``"most_probable_outcome"``.
            inner_solver_factory: Callable returning a (name, params) tuple
                specifying the inner solver and its parameters.
                Defaults to ``lambda: ("Astar", {})``.
            rho: Replanning probability threshold for convergence.
                Defaults to 0.1.
            mc_samples: Number of Monte-Carlo rollout samples per iteration.
                Defaults to 100.
            max_iterations: Maximum plan-merge iterations. Defaults to 50.
            max_steps: Maximum steps per MC rollout. Defaults to 10000.
            dead_end_cost: Cost for dead-end terminal states. Defaults to 1e9.
            optimize_policy_graph: Run discounted value iteration on policy
                graph after each plan merge. Defaults to False.
            discount: Discount factor for SSP optimization (< 1 for convergence
                with dead-end terminals). Defaults to 0.99.
            epsilon: Convergence threshold for value iteration. Defaults to 1e-3.
            parallel: Parallelize domain calls. Defaults to False.
            shared_memory_proxy: Optional shared memory proxy.
            continuous_planning: Re-solve from the current state on every
                call to get_next_action. Defaults to False.
            callback: Called after each iteration; return True to stop.
                Defaults to never stop.
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
            self._continuous_planning = continuous_planning
            self._lambdas = [heuristic]
            self._ipc_notify = True

            self._solver = sspplanmerger_solver(
                solver=self,
                domain=self.get_domain(),
                goal_checker=lambda d, s: d.is_goal(s),
                heuristic=lambda d, s: heuristic(d, s),
                determinization=determinization,
                inner_solver=inner_solver,
                inner_solver_params=inner_solver_params,
                rho=rho,
                mc_samples=mc_samples,
                max_iterations=max_iterations,
                max_steps=max_steps,
                dead_end_cost=dead_end_cost,
                optimize_policy_graph=optimize_policy_graph,
                discount=discount,
                epsilon=epsilon,
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
            """Run the plan-merging algorithm from a given state.

            # Parameters
            memory: State from which to start planning.
            """
            self._solver.solve(memory)

        def _resolve_from(self, memory: D.T_memory[D.T_state]) -> None:
            """Extend the existing policy from a given state without clearing.

            # Parameters
            memory: State from which to continue planning.
            """
            self._solver.resolve(memory)

        def _is_solution_defined_for(
            self, observation: D.T_agent[D.T_observation]
        ) -> bool:
            return self._solver.is_solution_defined_for(observation)

        def _get_next_action(
            self,
            observation: D.T_agent[D.T_observation],
            domain: Optional[Domain] = None,
        ) -> D.T_agent[D.T_concurrency[D.T_event]]:
            if self._continuous_planning or not self._is_solution_defined_for(
                observation
            ):
                self._resolve_from(observation)
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

        def get_nb_iterations(self) -> int:
            """Get the number of plan-merge iterations performed."""
            return self._solver.get_nb_iterations()

        def get_nb_plans(self) -> int:
            """Get the total number of deterministic plans computed."""
            return self._solver.get_nb_plans()

        def get_solving_time(self) -> int:
            """Get the total solving time in milliseconds."""
            return self._solver.get_solving_time()

        def get_policy_size(self) -> int:
            """Get the number of states in the policy."""
            return self._solver.get_policy_size()

        def get_explored_states(self) -> set:
            """Get the set of explored states in the policy."""
            return self._solver.get_explored_states()

        def get_terminal_states(self) -> set:
            """Get the set of terminal states (reachable from policy but not in policy and not goals)."""
            return self._solver.get_terminal_states()

        def get_policy(self) -> dict:
            """Get the full policy as a dict mapping state -> (action, value)."""
            return self._solver.get_policy()

except ImportError:
    print(
        "Scikit-decide C++ hub library not found. Please check it is "
        'installed in "skdecide/hub".'
    )
    raise
