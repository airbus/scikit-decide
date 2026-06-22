# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable
from typing import Optional, TypedDict

from skdecide import Distribution, Domain, Solver, Value
from skdecide.builders.domain import (
    Actions,
    EnumerableTransitions,
    Goals,
    Markovian,
    PartiallyObservable,
    PositiveCosts,
    Rewards,
    Sequential,
    SingleAgent,
    UncertainInitialized,
)
from skdecide.builders.solver import DeterministicPolicies, FromAnyState, Utilities
from skdecide.builders.solver.parallelability import ParallelSolver

try:
    from skdecide.hub.__skdecide_hub_cpp import _GoalHSVISolver_ as goal_hsvi_solver
    from skdecide.hub.__skdecide_hub_cpp import _HSVISolver_ as hsvi_solver

    class D_reward(
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

    class D_cost(
        Domain,
        SingleAgent,
        Sequential,
        EnumerableTransitions,
        Actions,
        Goals,
        Markovian,
        PartiallyObservable,
        PositiveCosts,
        UncertainInitialized,
    ):
        pass

    class HSVI(ParallelSolver, Solver, DeterministicPolicies, Utilities, FromAnyState):
        """HSVI solver for discounted reward POMDPs.

        From: Smith & Simmons, "Heuristic Search Value Iteration for POMDPs",
        UAI 2004.

        HSVI maintains dual bounds on the value function:
        - Lower bound: set of alpha-vectors, V(b) = max_alpha (alpha . b)
        - Upper bound: sawtooth interpolation from MDP corner values

        It performs heuristic search in belief space, selecting actions via
        the upper bound (optimistic) and observations via excess uncertainty.
        Converges when the gap at the initial belief falls below epsilon.

        Uses reward maximization with discount factor < 1.
        """

        T_domain = D_reward

        class BeliefDict(TypedDict):
            """Type for belief dictionaries returned by get_last_trajectory().

            Fields:
            - state_probs: list of (state, probability) tuples representing the belief distribution
            """

            state_probs: list[tuple[D_reward.T_state, float]]

        class AlphaVectorDict(TypedDict):
            """Type for alpha vector dictionaries returned by get_alpha_vectors().

            Fields:
            - values: dict mapping states (D.T_state) to Value[D.T_value]
            - action: action object (D.T_agent[D.T_concurrency[D.T_event]])
            - id: unique identifier for this alpha-vector
            """

            values: dict[D_reward.T_state, Value[D_reward.T_value]]
            action: D_reward.T_agent[D_reward.T_concurrency[D_reward.T_event]]
            id: int

        def __init__(
            self,
            domain_factory: Callable[[], Domain],
            epsilon: float = 0.001,
            discount: float = 0.95,
            time_budget: int = 300000,
            max_sample_depth: int = 100,
            max_vi_iterations: int = 1000,
            vi_convergence_factor: float = 0.01,
            belief_hash_resolution: float = 1000.0,
            parallel: bool = False,
            terminal_value: Callable[
                [D_reward.T_state], Value[D_reward.T_value]
            ] = lambda s: Value(reward=0),
            callback: Callable[[HSVI], bool] = lambda slv: False,
            verbose: bool = False,
        ) -> None:
            """Construct an HSVI solver instance.

            # Parameters
            domain_factory: Lambda function to create a domain instance.
            epsilon: Convergence threshold for the gap V_upper(b0) - V_lower(b0).
                Defaults to 0.001.
            discount: Discount factor gamma (must be < 1). Defaults to 0.95.
            time_budget: Maximum solving time in milliseconds. Defaults to 300000.
            max_sample_depth: Maximum depth for heuristic exploration.
                Defaults to 100.
            max_vi_iterations: Maximum iterations for bound initialization VI.
                Defaults to 1000.
            vi_convergence_factor: Convergence factor for initialization VI.
                Defaults to 0.01.
            belief_hash_resolution: Discretization factor for belief hashing.
                Probabilities are multiplied by this value and rounded to
                integers for hash computation. Defaults to 1000.0.
            parallel: Whether to use parallel C++ computation. Defaults to False.
            terminal_value: Optional function (state) -> value for terminal
                states. Defaults to 0.0.
            callback: Function called at each iteration. Return True to stop.
                Defaults to never stop.
            verbose: Whether to log progress messages. Defaults to False.
            """
            Solver.__init__(self, domain_factory=domain_factory)
            ParallelSolver.__init__(self, parallel=parallel)
            self._ipc_notify = True

            self._solver = hsvi_solver(
                solver=self,
                domain=self.get_domain(),
                epsilon=epsilon,
                discount=discount,
                time_budget=time_budget,
                max_sample_depth=max_sample_depth,
                use_closed_list=False,
                depth_bound_eta=0.1,
                max_vi_iterations=max_vi_iterations,
                vi_convergence_factor=vi_convergence_factor,
                belief_hash_resolution=belief_hash_resolution,
                parallel=parallel,
                terminal_value=terminal_value,
                callback=callback,
                verbose=verbose,
            )

        def close(self):
            if self._parallel:
                self._solver.close()
            ParallelSolver.close(self)

        def _solve(self, from_memory=None) -> None:
            if from_memory is None:
                tmp_domain = self._domain_factory()
                from_memory = tmp_domain.get_initial_state_distribution()
            self._solve_from(from_memory)

        def _solve_from(self, initial_belief: Distribution[D_reward.T_state]) -> None:
            self._solver.solve(initial_belief)

        def _is_solution_defined_for(
            self, observation: D_reward.T_agent[D_reward.T_observation]
        ) -> bool:
            return self._solver.is_solution_defined_for(observation)

        def _get_next_action(
            self,
            observation: D_reward.T_agent[D_reward.T_observation],
            domain: Optional[Domain] = None,
        ) -> D_reward.T_agent[D_reward.T_concurrency[D_reward.T_event]]:
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

        def _get_utility(
            self, observation: D_reward.T_agent[D_reward.T_observation]
        ) -> D_reward.T_value:
            return self._solver.get_utility(observation)

        def get_next_action_from_belief(
            self, belief: Distribution[D_reward.T_state]
        ) -> D_reward.T_agent[D_reward.T_concurrency[D_reward.T_event]]:
            """Get the best action for an explicit belief state."""
            action = self._solver.get_next_action_from_belief(belief)
            if action is None:
                print(
                    "\x1b[3;33;40m"
                    + "No best action found for belief, applying random action"
                    + "\x1b[0m"
                )
                return self.call_domain_method("get_action_space").sample()
            else:
                return action

        def get_utility_from_belief(
            self, belief: Distribution[D_reward.T_state]
        ) -> Value[D_reward.T_value]:
            """Get the best value for an explicit belief state."""
            return self._solver.get_utility_from_belief(belief)

        def is_solution_defined_for_from_belief(
            self, belief: Distribution[D_reward.T_state]
        ) -> bool:
            """Check if a solution is defined for an explicit belief state."""
            return self._solver.is_solution_defined_for_from_belief(belief)

        def reset_belief(self) -> None:
            """Reset the tracked belief to the initial belief from solve()."""
            self._solver.reset_belief()

        def get_nb_alpha_vectors(self) -> int:
            """Get the number of alpha-vectors in the lower bound."""
            return self._solver.get_nb_alpha_vectors()

        def get_nb_bound_points(self) -> int:
            """Get the number of interior points in the upper bound."""
            return self._solver.get_nb_bound_points()

        def get_solving_time(self) -> int:
            """Get the solving time in milliseconds."""
            return self._solver.get_solving_time()

        def get_gap(self) -> float:
            """Get the current gap V_upper(b0) - V_lower(b0)."""
            return self._solver.get_gap()

        def get_alpha_vectors(self) -> list[AlphaVectorDict]:
            """Get the alpha-vectors representing the lower bound.

            Returns a list of dictionaries, each containing:
            - 'values': dict[D.T_state, Value[D.T_value]] - state to Value mapping
            - 'action': D.T_agent[D.T_concurrency[D.T_event]] - associated action
            - 'id': int - unique identifier for this alpha-vector

            The lower bound at any belief b is: V_lower(b) = max_alpha (alpha · b),
            where alpha · b = sum(b[s] * alpha['values'][s].reward for all states s).
            The policy chooses the action of the maximizing alpha-vector.
            """
            return self._solver.get_alpha_vectors()

        def get_last_trajectory(
            self,
        ) -> list[
            tuple[
                "HSVI.BeliefDict",
                D_reward.T_agent[D_reward.T_concurrency[D_reward.T_event]],
            ]
        ]:
            """Get the ordered list of (belief, action) pairs visited during
            the last HSVI exploration.

            Returns the trajectory (path) explored during the most recent explore()
            call. Each element is a tuple of (belief_dict, action) where belief_dict
            contains 'state_probs': a list of (state, probability) tuples, and action
            is the greedy action (optimistic, via upper bound) selected at that belief.

            Note: HSVI operates on continuous belief spaces via heuristic search.
            Beliefs are returned as dictionaries with state-probability mappings.

            # Returns
            list[tuple[HSVI.BeliefDict, object]]: List of (belief, action) pairs
                visited during the last exploration. Returns an empty list if solve()
                has not been called yet.
            """

            return self._solver.get_last_trajectory()

    class GoalHSVI(
        ParallelSolver, Solver, DeterministicPolicies, Utilities, FromAnyState
    ):
        """Goal-HSVI solver for undiscounted cost Goal-POMDPs.

        From: Horak, Bosansky, Chatterjee, "Goal-HSVI: Heuristic Search
        Value Iteration for Goal-POMDPs", IJCAI 2018.

        Goal-HSVI extends HSVI to handle undiscounted Goal-POMDPs with
        cost minimization. Key modifications over vanilla HSVI:
        - Bounds are swapped: alpha-vectors form the upper bound (min),
          sawtooth interpolation forms the lower bound
        - Bounded search depth T based on cost bounds
        - Closed list to avoid re-exploring same beliefs

        Uses cost minimization with goals. Goal states have value 0;
        non-goal terminal states (dead ends) receive a large dead-end cost.
        """

        T_domain = D_cost

        class BeliefDict(TypedDict):
            """Type for belief dictionaries returned by get_last_trajectory().

            Fields:
            - state_probs: list of (state, probability) tuples representing the belief distribution
            """

            state_probs: list[tuple[D_cost.T_state, float]]

        class AlphaVectorDict(TypedDict):
            """Type for alpha vector dictionaries returned by get_alpha_vectors().

            Fields:
            - values: dict mapping states (D.T_state) to Value[D.T_value]
            - action: action object (D.T_agent[D.T_concurrency[D.T_event]])
            - id: unique identifier for this alpha-vector
            """

            values: dict[D_cost.T_state, Value[D_cost.T_value]]
            action: D_cost.T_agent[D_cost.T_concurrency[D_cost.T_event]]
            id: int

        def __init__(
            self,
            domain_factory: Callable[[], Domain],
            goal_checker: Callable[[Domain, D_cost.T_state], bool] = lambda d,
            s: d.is_goal(s),
            epsilon: float = 0.001,
            time_budget: int = 300000,
            max_sample_depth: int = 100,
            depth_bound_eta: float = 0.1,
            max_vi_iterations: int = 1000,
            vi_convergence_factor: float = 0.01,
            belief_hash_resolution: float = 1000.0,
            parallel: bool = False,
            terminal_value: Optional[
                Callable[[D_cost.T_state], Value[D_cost.T_value]]
            ] = None,
            callback: Callable[[GoalHSVI], bool] = lambda slv: False,
            verbose: bool = False,
            dead_end_cost: Optional[float] = None,
        ) -> None:
            """Construct a Goal-HSVI solver instance.

            # Parameters
            domain_factory: Lambda function to create a domain instance.
            goal_checker: Function (domain, state) -> bool that returns True
                if the state is a goal. Defaults to domain.is_goal(state).
            epsilon: Convergence threshold for the gap V_upper(b0) - V_lower(b0).
                Defaults to 0.001.
            time_budget: Maximum solving time in milliseconds. Defaults to 300000.
            max_sample_depth: Maximum depth for heuristic exploration.
                Also used as fallback when depth bound cannot be computed.
                Defaults to 100.
            depth_bound_eta: Parameter eta for depth bound computation:
                T = ceil(C_max/c_min * (C_max - eta*eps) / ((1-eta)*eps)).
                Defaults to 0.1.
            max_vi_iterations: Maximum iterations for bound initialization VI.
                Defaults to 1000.
            vi_convergence_factor: Convergence factor for initialization VI.
                Defaults to 0.01.
            belief_hash_resolution: Discretization factor for belief hashing.
                Probabilities are multiplied by this value and rounded to
                integers for hash computation. Defaults to 1000.0.
            parallel: Whether to use parallel C++ computation. Defaults to False.
            terminal_value: Optional function (state) -> value for terminal
                states. Overrides goal_checker + dead_end_cost logic if provided.
                Defaults to None (use goal_checker + dead_end_cost logic).
            callback: Function called at each iteration. Return True to stop.
                Defaults to never stop.
            verbose: Whether to log progress messages. Defaults to False.
            dead_end_cost: Cost assigned to non-goal terminal states (dead ends).
                If None (default), automatically computed as
                max_transition_cost * max_sample_depth (undiscounted) or
                max_transition_cost / (1 - discount) (discounted).
                Ignored if terminal_value is provided.
            """
            Solver.__init__(self, domain_factory=domain_factory)
            ParallelSolver.__init__(self, parallel=parallel)
            self._ipc_notify = True
            self._lambdas = [goal_checker]

            self._solver = goal_hsvi_solver(
                solver=self,
                domain=self.get_domain(),
                goal_checker=goal_checker,
                epsilon=epsilon,
                discount=1.0,
                time_budget=time_budget,
                max_sample_depth=max_sample_depth,
                use_closed_list=True,
                depth_bound_eta=depth_bound_eta,
                max_vi_iterations=max_vi_iterations,
                vi_convergence_factor=vi_convergence_factor,
                belief_hash_resolution=belief_hash_resolution,
                parallel=parallel,
                terminal_value=terminal_value,
                callback=callback,
                verbose=verbose,
                dead_end_cost=dead_end_cost,
            )

        def close(self):
            if self._parallel:
                self._solver.close()
            ParallelSolver.close(self)

        def _solve(self, from_memory=None) -> None:
            if from_memory is None:
                tmp_domain = self._domain_factory()
                from_memory = tmp_domain.get_initial_state_distribution()
            self._solve_from(from_memory)

        def _solve_from(self, initial_belief: Distribution[D_cost.T_state]) -> None:
            self._solver.solve(initial_belief)

        def _is_solution_defined_for(
            self, observation: D_cost.T_agent[D_cost.T_observation]
        ) -> bool:
            return self._solver.is_solution_defined_for(observation)

        def _get_next_action(
            self,
            observation: D_cost.T_agent[D_cost.T_observation],
            domain: Optional[Domain] = None,
        ) -> D_cost.T_agent[D_cost.T_concurrency[D_cost.T_event]]:
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

        def _get_utility(
            self, observation: D_cost.T_agent[D_cost.T_observation]
        ) -> D_cost.T_value:
            return self._solver.get_utility(observation)

        def get_next_action_from_belief(
            self, belief: Distribution[D_cost.T_state]
        ) -> D_cost.T_agent[D_cost.T_concurrency[D_cost.T_event]]:
            """Get the best action for an explicit belief state."""
            action = self._solver.get_next_action_from_belief(belief)
            if action is None:
                print(
                    "\x1b[3;33;40m"
                    + "No best action found for belief, applying random action"
                    + "\x1b[0m"
                )
                return self.call_domain_method("get_action_space").sample()
            else:
                return action

        def get_utility_from_belief(
            self, belief: Distribution[D_cost.T_state]
        ) -> Value[D_cost.T_value]:
            """Get the best value for an explicit belief state."""
            return self._solver.get_utility_from_belief(belief)

        def is_solution_defined_for_from_belief(
            self, belief: Distribution[D_cost.T_state]
        ) -> bool:
            """Check if a solution is defined for an explicit belief state."""
            return self._solver.is_solution_defined_for_from_belief(belief)

        def reset_belief(self) -> None:
            """Reset the tracked belief to the initial belief from solve()."""
            self._solver.reset_belief()

        def get_nb_alpha_vectors(self) -> int:
            """Get the number of alpha-vectors in the upper bound."""
            return self._solver.get_nb_alpha_vectors()

        def get_nb_bound_points(self) -> int:
            """Get the number of interior points in the lower bound."""
            return self._solver.get_nb_bound_points()

        def get_solving_time(self) -> int:
            """Get the solving time in milliseconds."""
            return self._solver.get_solving_time()

        def get_gap(self) -> float:
            """Get the current gap V_upper(b0) - V_lower(b0)."""
            return self._solver.get_gap()

        def get_alpha_vectors(self) -> list[AlphaVectorDict]:
            """Get the alpha-vectors representing the upper bound.

            Returns a list of dictionaries, each containing:
            - 'values': dict[D.T_state, Value[D.T_value]] - state to Value mapping
            - 'action': D.T_agent[D.T_concurrency[D.T_event]] - associated action
            - 'id': int - unique identifier for this alpha-vector

            Note: In Goal-HSVI (cost minimization), alpha-vectors form the UPPER bound,
            unlike vanilla HSVI where they form the lower bound. The upper bound at any
            belief b is: V_upper(b) = min_alpha (alpha · b), where
            alpha · b = sum(b[s] * alpha['values'][s].cost for all states s).
            The policy chooses the action of the minimizing alpha-vector.
            """
            return self._solver.get_alpha_vectors()

        def get_last_trajectory(
            self,
        ) -> list[
            tuple[
                "GoalHSVI.BeliefDict",
                D_cost.T_agent[D_cost.T_concurrency[D_cost.T_event]],
            ]
        ]:
            """Get the ordered list of (belief, action) pairs visited during
            the last Goal-HSVI exploration.

            Returns the trajectory (path) explored during the most recent explore()
            call. Each element is a tuple of (belief_dict, action) where belief_dict
            contains 'state_probs': a list of (state, probability) tuples, and action
            is the greedy action (pessimistic in cost minimization, via upper bound)
            selected at that belief.

            Note: Goal-HSVI operates on continuous belief spaces with bounded depth search.
            Beliefs are returned as dictionaries with state-probability mappings.

            # Returns
            list[tuple[GoalHSVI.BeliefDict, object]]: List of (belief, action) pairs
                visited during the last exploration. Returns an empty list if solve()
                has not been called yet.
            """

            return self._solver.get_last_trajectory()

except ImportError:
    print(
        'Scikit-decide C++ hub library not found. Please check it is installed in "skdecide/hub".'
    )
    raise
