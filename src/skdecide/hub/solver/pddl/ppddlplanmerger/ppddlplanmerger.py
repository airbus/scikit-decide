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
from skdecide.builders.solver import DeterministicPolicies, Utilities

try:
    from skdecide.hub.__skdecide_hub_cpp import (
        _PPDDLPlanMergerSolver_ as CppPPDDLPlanMergerSolver,
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

    class PPDDLPlanMerger(Solver, DeterministicPolicies, Utilities):
        """Plan-merging solver for probabilistic PDDL (PPDDL) domains
        with a pluggable inner deterministic solver.

        Iteratively determinizes the stochastic domain at the PDDL
        effect-tree level, plans from terminal states, and merges plans
        into a policy until Monte-Carlo assessment shows the replanning
        probability is below a threshold rho.

        Counterpart of RFF, but supports pluggable inner solvers via name.

        Three determinization strategies:
        - "most_probable_outcome": use highest-probability outcome
        - "all_outcomes": expand each stochastic action into N deterministic
          actions
        - "random_outcome": sample one outcome randomly (re-sampled each plan)

        Optional discounted value iteration on the policy graph treats
        terminal states as absorbing dead-ends and optimizes actions.

        Requires a PPDDLDomain.

        # Reference
        Teichteil-Königsbuch, F., Kuter, U., & Infantes, G. (2010).
        RFF: A Robust, FF-Based MDP Planning Algorithm for Generating
        Policies with Low Probability of Failure. In *Proc. AAMAS*.
        """

        T_domain = D

        def __init__(
            self,
            domain_factory: Callable[[], Domain],
            inner_solver_factory: Optional[Callable[[], tuple[str, dict]]] = None,
            determinization: str = "most_probable_outcome",
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
            callback: Callable[["PPDDLPlanMerger"], bool] = lambda slv: False,
            verbose: bool = False,
        ) -> None:
            """Construct a PPDDLPlanMerger solver instance.

            # Parameters
            domain_factory: Lambda to create a PPDDL domain instance.
            inner_solver_factory: Factory returning (solver_name, params_dict).
                Defaults to ("FF", {}). Use get_available_pddl_inner_solvers()
                to list registered solvers.
            determinization: Determinization strategy. One of
                "most_probable_outcome", "all_outcomes", or
                "random_outcome". Defaults to "most_probable_outcome".
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
            continuous_planning: Re-solve from the current state on every
                call to get_next_action. Defaults to False.
            parallel: Parallelize domain evaluation. Defaults to False.
            callback: Called after each iteration; return True to stop.
                Defaults to never stop.
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
            self._determinization = determinization
            self._rho = rho
            self._mc_samples = mc_samples
            self._max_iterations = max_iterations
            self._max_steps = max_steps
            self._dead_end_cost = dead_end_cost
            self._optimize_policy_graph = optimize_policy_graph
            self._discount = discount
            self._epsilon = epsilon
            self._continuous_planning = continuous_planning
            self._parallel = parallel
            self._callback = callback
            self._verbose = verbose

        def _solve(self) -> None:
            domain = self._domain_factory()
            if not hasattr(domain, "_task"):
                raise TypeError(
                    "PPDDLPlanMerger solver requires a PPDDLDomain "
                    "(with _task attribute)"
                )
            self._task = domain._task
            self._cpp_solver = CppPPDDLPlanMergerSolver(
                self,
                self._task,
                self._inner_solver_name,
                self._determinization,
                self._parallel,
                self._dead_end_cost,
                self._rho,
                self._mc_samples,
                self._max_iterations,
                self._max_steps,
                self._optimize_policy_graph,
                self._discount,
                self._epsilon,
                self._callback,
                self._verbose,
                self._inner_solver_params,
            )
            self._cpp_solver.solve(self._task.initial_state())

        def _resolve_from(self, observation: D.T_agent[D.T_observation]) -> None:
            self._cpp_solver.resolve(observation.to_cpp())

        def _is_solution_defined_for(
            self, observation: D.T_agent[D.T_observation]
        ) -> bool:
            return self._cpp_solver.is_solution_defined_for(observation.to_cpp())

        def _get_next_action(
            self,
            observation: D.T_agent[D.T_observation],
            domain: Optional[Domain] = None,
        ) -> D.T_agent[D.T_concurrency[D.T_event]]:
            if self._continuous_planning or not self._is_solution_defined_for(
                observation
            ):
                self._resolve_from(observation)
            action = self._cpp_solver.get_next_action(observation.to_cpp())
            if action is None:
                return None
            return PDDLAction(action, self._task)

        def _get_utility(self, observation: D.T_agent[D.T_observation]) -> D.T_value:
            return self._cpp_solver.get_best_value(observation.to_cpp())

        def get_nb_iterations(self) -> int:
            """Get the number of plan-merge iterations performed."""
            return self._cpp_solver.get_nb_iterations()

        def get_nb_plans(self) -> int:
            """Get the total number of deterministic plans computed."""
            return self._cpp_solver.get_nb_plans()

        def get_solving_time(self) -> int:
            """Get the total solving time in milliseconds."""
            return self._cpp_solver.get_solving_time()

        def get_policy_size(self) -> int:
            """Get the number of states in the policy."""
            return self._cpp_solver.get_policy_size()

        def get_explored_states(self) -> set:
            """Get the set of explored states in the policy."""
            return self._cpp_solver.get_explored_states()

        def get_terminal_states(self) -> set:
            """Get the set of terminal states (reachable from policy but not in policy and not goals)."""
            return self._cpp_solver.get_terminal_states()

        def get_policy(self) -> dict:
            """Get the full policy as a dict mapping state -> (action, value)."""
            return self._cpp_solver.get_policy()

except ImportError:
    print(
        "Scikit-decide C++ hub library not found. Please check it is "
        'installed in "skdecide/hub".'
    )
    raise
