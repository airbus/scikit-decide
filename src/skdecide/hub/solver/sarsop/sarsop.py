# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

from skdecide import Distribution, Domain, Solver
from skdecide.builders.domain import (
    Actions,
    EnumerableTransitions,
    Markovian,
    PartiallyObservable,
    Rewards,
    Sequential,
    SingleAgent,
    UncertainInitialized,
)
from skdecide.builders.solver import (
    DeterministicPolicies,
    FromAnyState,
    ParallelSolver,
    Utilities,
)

try:
    from skdecide.hub.__skdecide_hub_cpp import _SARSOPSolver_ as sarsop_solver

    class D(
        Domain,
        SingleAgent,
        Sequential,
        EnumerableTransitions,
        Actions,
        Markovian,
        PartiallyObservable,
        Rewards,
        UncertainInitialized,
    ):
        pass

    class SARSOP(
        ParallelSolver, Solver, DeterministicPolicies, Utilities, FromAnyState
    ):
        """SARSOP solver for general POMDPs (reward maximization).

        From: Kurniawati, Hsu & Lee, "SARSOP: Efficient Point-Based POMDP
        Planning by Approximating Optimally Reachable Belief Spaces", RSS 2008.

        SARSOP maintains dual bounds (lower via alpha-vectors, upper via
        sawtooth interpolation) and uses guided exploration of the belief
        tree focused on the optimally reachable belief space.

        The default interface works with observations. The solver internally
        maintains and updates the current belief using Bayes rule from the
        observation history.
        """

        T_domain = D

        def __init__(
            self,
            domain_factory: Callable[[], Domain],
            epsilon: float = 0.001,
            discount: float = 0.95,
            time_budget: int = 300000,
            max_beliefs: int = 100000,
            pruning_delta: float = 1e-6,
            max_vi_iterations: int = 1000,
            vi_convergence_factor: float = 0.01,
            max_sample_depth: int = 100,
            prob_epsilon: float = 1e-15,
            ub_improvement_epsilon: float = 1e-10,
            pruning_interval: int = 10,
            logging_interval: int = 50,
            parallel: bool = False,
            shared_memory_proxy=None,
            callback: Callable[[SARSOP], bool] = lambda slv: False,
            verbose: bool = False,
        ) -> None:
            """Construct a SARSOP solver instance.

            # Parameters
            domain_factory: Lambda function to create a domain instance.
            epsilon: Convergence threshold for the gap V_upper(b0) - V_lower(b0).
                Defaults to 0.001.
            discount: Discount factor gamma. Must be in (0, 1).
                Defaults to 0.95.
            time_budget: Maximum solving time in milliseconds.
                Defaults to 300000 (5 minutes).
            max_beliefs: Maximum number of belief tree nodes.
                Defaults to 100000.
            pruning_delta: Delta parameter for alpha-vector dominance pruning.
                Defaults to 1e-6.
            max_vi_iterations: Maximum iterations for bound initialization VI.
                Defaults to 1000.
            vi_convergence_factor: VI convergence threshold is
                epsilon * vi_convergence_factor. Defaults to 0.01.
            max_sample_depth: Maximum depth for belief tree sampling.
                Defaults to 100.
            prob_epsilon: Near-zero probability threshold. Defaults to 1e-15.
            ub_improvement_epsilon: Minimum upper-bound improvement to record
                an interior point. Defaults to 1e-10.
            pruning_interval: Number of iterations between alpha-vector
                pruning passes. Set to 0 to disable. Defaults to 10.
            logging_interval: Number of iterations between verbose log
                messages. Set to 0 to disable. Defaults to 50.
            parallel: Parallelize domain calls. Defaults to False.
            shared_memory_proxy: Optional shared memory proxy. Defaults to None.
            callback: Function called at end of each iteration, taking the
                solver as argument, returning True to stop. Defaults to
                never stop.
            verbose: Whether to log verbose messages. Defaults to False.
            """
            Solver.__init__(self, domain_factory=domain_factory)
            ParallelSolver.__init__(
                self,
                parallel=parallel,
                shared_memory_proxy=shared_memory_proxy,
            )
            self._ipc_notify = True

            self._solver = sarsop_solver(
                solver=self,
                domain=self.get_domain(),
                epsilon=epsilon,
                discount=discount,
                time_budget=time_budget,
                max_beliefs=max_beliefs,
                pruning_delta=pruning_delta,
                max_vi_iterations=max_vi_iterations,
                vi_convergence_factor=vi_convergence_factor,
                max_sample_depth=max_sample_depth,
                prob_epsilon=prob_epsilon,
                ub_improvement_epsilon=ub_improvement_epsilon,
                pruning_interval=pruning_interval,
                logging_interval=logging_interval,
                parallel=parallel,
                callback=callback,
                verbose=verbose,
            )

        def close(self):
            """Joins the parallel domains' processes."""
            if self._parallel:
                self._solver.close()
            ParallelSolver.close(self)

        def _solve(self, from_memory=None) -> None:
            if from_memory is None:
                from_memory = self._domain.get_initial_state_distribution()
            self._solve_from(from_memory)

        def _solve_from(self, initial_belief: Distribution[D.T_state]) -> None:
            """Run SARSOP from an initial belief distribution.

            # Parameters
            initial_belief: Distribution over physical states representing
                the initial belief.
            """
            self._solver.solve(initial_belief)

        def _is_solution_defined_for(
            self, observation: D.T_agent[D.T_observation]
        ) -> bool:
            return self._solver.is_solution_defined_for(observation)

        def _get_next_action(
            self,
            observation: D.T_agent[D.T_observation],
            domain: Optional[Domain] = None,
        ) -> D.T_agent[D.T_concurrency[D.T_event]]:
            """Get the best action given an observation.

            The solver internally maintains and updates the current belief
            using the last action returned and the new observation via Bayes
            rule. On the first call after solve(), the initial belief is used.
            """
            action = self._solver.get_next_action(observation)
            if action is None:
                print(
                    "\x1b[3;33;40m"
                    + "No best action found for observation "
                    + str(observation)
                    + ", applying random action"
                    + "\x1b[0m"
                )
                return self.call_domain_method("get_action_space").sample()
            else:
                return action

        def _get_utility(self, observation: D.T_agent[D.T_observation]) -> D.T_value:
            return self._solver.get_utility(observation)

        def get_next_action_from_belief(
            self, belief: Distribution[D.T_state]
        ) -> D.T_agent[D.T_concurrency[D.T_event]]:
            """Get the best action for an explicit belief state."""
            action = self._solver.get_next_action_from_belief(belief)
            if action is None:
                print(
                    "\x1b[3;33;40m"
                    + "No best action found for belief, applying random action"
                    + "\x1b[0m"
                )
                return self.call_domain_method("get_action_space").sample()
            return action

        def get_utility_from_belief(self, belief: Distribution[D.T_state]) -> D.T_value:
            """Get the best value for an explicit belief state."""
            return self._solver.get_utility_from_belief(belief)

        def is_solution_defined_for_from_belief(
            self, belief: Distribution[D.T_state]
        ) -> bool:
            """Check if a solution is defined for an explicit belief state."""
            return self._solver.is_solution_defined_for_from_belief(belief)

        def reset_belief(self) -> None:
            """Reset the tracked belief to the initial belief from solve()."""
            self._solver.reset_belief()

        def get_nb_alpha_vectors(self) -> int:
            """Get the number of alpha-vectors in the lower bound."""
            return self._solver.get_nb_alpha_vectors()

        def get_nb_explored_beliefs(self) -> int:
            """Get the number of belief tree nodes."""
            return self._solver.get_nb_explored_beliefs()

        def get_solving_time(self) -> int:
            """Get the solving time in milliseconds."""
            return self._solver.get_solving_time()

        def get_lower_bound(self) -> float:
            """Get V_lower(b0) at the root belief."""
            return self._solver.get_lower_bound()

        def get_upper_bound(self) -> float:
            """Get V_upper(b0) at the root belief."""
            return self._solver.get_upper_bound()

        def get_gap(self) -> float:
            """Get the gap V_upper(b0) - V_lower(b0)."""
            return self._solver.get_gap()

except ImportError:
    print(
        "Scikit-decide C++ hub library not found. Please check it is installed "
        'in "skdecide/hub".'
    )
    raise
