# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

from skdecide import Distribution, Domain, Solver, Value
from skdecide.builders.domain import (
    Actions,
    Markovian,
    PartiallyObservable,
    Rewards,
    Sequential,
    SingleAgent,
    UncertainInitialized,
    UncertainTransitions,
)
from skdecide.builders.solver import (
    DeterministicPolicies,
    FromAnyState,
    ParallelSolver,
    Utilities,
)

try:
    from skdecide.hub.__skdecide_hub_cpp import _DespotSolver_ as despot_solver

    class D(
        Domain,
        SingleAgent,
        Sequential,
        UncertainTransitions,
        Actions,
        Markovian,
        PartiallyObservable,
        Rewards,
        UncertainInitialized,
    ):
        pass

    class DESPOT(
        ParallelSolver, Solver, DeterministicPolicies, Utilities, FromAnyState
    ):
        """DESPOT solver for POMDPs (online, anytime, reward maximization).

        From: Ye, Somani, Hsu & Lee, "DESPOT: Online POMDP Planning with
        Regularization", JAIR 2017.

        DESPOT builds a sparse AND-OR tree using K determinized scenarios
        sampled from the current belief. Regularization (RWDU) prevents
        overfitting to the sampled scenarios. Planning happens online at
        each step within a time budget.

        The default interface works with observations. The solver internally
        maintains and updates the current belief using a particle filter.
        """

        T_domain = D

        def __init__(
            self,
            domain_factory: Callable[[], Domain],
            num_scenarios: int = 500,
            max_depth: int = 90,
            regularization_constant: float = 0.0,
            gap_reduction_rate: float = 0.95,
            target_gap: float = 0.0,
            time_budget: int = 1000,
            discount: float = 0.95,
            max_rollout_depth: int = 90,
            num_particles_belief_update: int = 500,
            ess_threshold_ratio: float = 2.0,
            default_policy: Optional[Callable[[Domain, object], Value]] = None,
            upper_bound_heuristic: Optional[Callable[[Domain, object], Value]] = None,
            parallel: bool = False,
            shared_memory_proxy=None,
            callback: Callable[[DESPOT, Optional[int]], bool] = lambda slv,
            i=None: False,
            verbose: bool = False,
        ) -> None:
            """Construct a DESPOT solver instance.

            # Parameters
            domain_factory: Lambda function to create a domain instance.
            num_scenarios: Number of determinized scenarios (K).
                Defaults to 500.
            max_depth: Maximum search depth in the DESPOT tree (D).
                Defaults to 90.
            regularization_constant: Regularization constant (lambda) for
                RWDU pruning. 0 disables regularization. Defaults to 0.0.
            gap_reduction_rate: Rate at which the target gap shrinks each
                iteration (xi). Defaults to 0.95.
            target_gap: Convergence threshold for the gap between upper
                and lower bounds (epsilon_0). Defaults to 0.0.
            time_budget: Maximum planning time per step in milliseconds.
                Defaults to 1000.
            discount: Discount factor gamma. Must be in (0, 1).
                Defaults to 0.95.
            max_rollout_depth: Maximum depth for default policy rollouts.
                Defaults to 90.
            num_particles_belief_update: Number of particles for belief
                update via particle filter. Defaults to 500.
            ess_threshold_ratio: Effective sample size threshold for
                resampling. Resampling occurs when ESS < N / ratio.
                Defaults to 2.0.
            default_policy: Optional function (domain, state) -> Value
                providing a lower bound via a default policy. If None,
                random rollouts are used. Defaults to None.
            upper_bound_heuristic: Optional function (domain, state) -> Value
                providing an upper bound heuristic. If None, R_max/(1-gamma)
                is used. Defaults to None.
            parallel: Parallelize domain calls. Defaults to False.
            shared_memory_proxy: Optional shared memory proxy.
                Defaults to None.
            callback: Function called at end of each iteration, taking the
                solver and an optional thread_id (int or None) as arguments,
                returning True to stop. Defaults to never stop.
            verbose: Whether to log verbose messages. Defaults to False.
            """
            Solver.__init__(self, domain_factory=domain_factory)
            ParallelSolver.__init__(
                self,
                parallel=parallel,
                shared_memory_proxy=shared_memory_proxy,
            )
            self._ipc_notify = True

            self._solver = despot_solver(
                solver=self,
                domain=self.get_domain(),
                num_scenarios=num_scenarios,
                max_depth=max_depth,
                regularization_constant=regularization_constant,
                gap_reduction_rate=gap_reduction_rate,
                target_gap=target_gap,
                time_budget=time_budget,
                discount=discount,
                max_rollout_depth=max_rollout_depth,
                num_particles_belief_update=num_particles_belief_update,
                ess_threshold_ratio=ess_threshold_ratio,
                default_policy=default_policy,
                upper_bound_heuristic=upper_bound_heuristic,
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
                from_memory = self._domain_factory().get_initial_state_distribution()
            self._solve_from(from_memory)

        def _solve_from(self, initial_belief: Distribution[D.T_state]) -> None:
            """Initialize DESPOT with an initial belief distribution.

            For DESPOT (an online solver), this only initializes the belief
            particles. Actual planning happens online in _get_next_action().

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

            The solver updates its belief via particle filter using the
            last action and new observation, then builds a DESPOT tree
            from the current belief within the time budget.
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

        def get_nb_tree_nodes(self) -> int:
            """Get the number of nodes in the last DESPOT tree."""
            return self._solver.get_nb_tree_nodes()

        def get_solving_time(self) -> int:
            """Get the last planning time in milliseconds."""
            return self._solver.get_solving_time()

        def get_gap(self) -> float:
            """Get the gap between upper and lower bounds at the root."""
            return self._solver.get_gap()

except ImportError:
    print(
        "Scikit-decide C++ hub library not found. Please check it is installed "
        'in "skdecide/hub".'
    )
    raise
