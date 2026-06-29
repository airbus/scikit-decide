# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    FloatHyperparameter,
    IntegerHyperparameter,
)

from skdecide import Distribution, Domain, Solver
from skdecide.builders.domain import (
    Actions,
    Goals,
    Markovian,
    PartiallyObservable,
    PositiveCosts,
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
from skdecide.core import Value

try:
    from skdecide.hub.__skdecide_hub_cpp import _RTDPBelSolver_ as rtdp_bel_solver

    class D(
        Domain,
        SingleAgent,
        Sequential,
        UncertainTransitions,
        Actions,
        Goals,
        Markovian,
        PartiallyObservable,
        PositiveCosts,
        UncertainInitialized,
    ):
        pass

    class RTDPBel(
        ParallelSolver, Solver, DeterministicPolicies, Utilities, FromAnyState
    ):
        """RTDP-Bel solver for Goal POMDPs.

        From: Bonet & Geffner, "Solving POMDPs: RTDP-Bel vs. Point-based
        Algorithms", IJCAI 2009.

        RTDP-Bel is RTDP applied to the belief MDP with discretized belief
        hashing. It performs trial-based search in belief space, where beliefs
        are probability distributions over physical states. Beliefs are
        discretized for hash table access using d(b(s)) = ceil(D * b(s)).

        The default interface works with observations. The solver internally
        maintains and updates the current belief using Bayes rule from the
        observation history.

        Uses cost minimization with goals over belief space.
        """

        T_domain = D

        hyperparameters = [
            IntegerHyperparameter(name="discretization"),
            IntegerHyperparameter(name="rollout_budget"),
            IntegerHyperparameter(name="max_depth"),
            FloatHyperparameter(name="epsilon"),
            FloatHyperparameter(name="discount"),
        ]

        def __init__(
            self,
            domain_factory: Callable[[], Domain],
            heuristic: Callable[
                [Domain, D.T_state], D.T_agent[Value[D.T_value]]
            ] = lambda d, s: Value(cost=0),
            terminal_value: Callable[
                [D.T_state], D.T_agent[Value[D.T_value]]
            ] = lambda s: Value(cost=0),
            discretization: int = 10,
            time_budget: int = 3600000,
            rollout_budget: int = 100000,
            max_depth: int = 1000,
            epsilon: float = 0.001,
            discount: float = 1.0,
            parallel: bool = False,
            shared_memory_proxy=None,
            callback: Callable[[RTDPBel, Optional[int]], bool] = lambda slv,
            i=None: False,
            verbose: bool = False,
        ) -> None:
            """Construct an RTDP-Bel solver instance.

            # Parameters
            domain_factory: Lambda function to create a domain instance.
            heuristic: Function h(domain, state) -> Value returning the
                heuristic cost estimate for a physical state. The belief
                heuristic is computed as h(b) = sum_s b(s)*h(s).
                Defaults to Value(cost=0).
            terminal_value: Function t(state) -> Value returning the value
                for terminal non-goal states (dead-ends). If None, defaults
                to Value(cost=0). For domains where terminal states are always
                goals, this can be left as None.
            discretization: Discretization parameter D for belief hashing.
                d(b(s)) = ceil(D * b(s)). Higher D = finer discretization
                but more memory. Defaults to 10.
            time_budget: Maximum solving time in milliseconds. Defaults to 3600000.
            rollout_budget: Maximum number of trials. Defaults to 100000.
            max_depth: Maximum depth of each trial. Defaults to 1000.
            epsilon: Maximum Bellman residual for convergence. Defaults to 0.001.
            discount: Value function's discount factor. Defaults to 1.0.
            parallel: Parallelize trials. Defaults to False.
            shared_memory_proxy: Optional shared memory proxy. Defaults to None.
            callback: Function called at the end of each trial with
                (solver, thread_id). thread_id is None when running
                sequentially. Defaults to never stop.
            verbose: Whether to log verbose messages. Defaults to False.
            """
            Solver.__init__(self, domain_factory=domain_factory)
            ParallelSolver.__init__(
                self,
                parallel=parallel,
                shared_memory_proxy=shared_memory_proxy,
            )
            self._lambdas = [heuristic]
            self._ipc_notify = True

            self._solver = rtdp_bel_solver(
                solver=self,
                domain=self.get_domain(),
                goal_checker=lambda d, s: d.is_goal(s),
                heuristic=(
                    (lambda d, s: heuristic(d, s))
                    if not parallel
                    else (lambda d, s: d.call(None, 0, s))
                ),
                terminal_value=(
                    (lambda s: terminal_value(s)) if terminal_value else None
                ),
                discretization=discretization,
                time_budget=time_budget,
                rollout_budget=rollout_budget,
                max_depth=max_depth,
                epsilon=epsilon,
                discount=discount,
                parallel=parallel,
                callback=callback,
                verbose=verbose,
            )

        def close(self):
            if self._parallel:
                self._solver.close()
            ParallelSolver.close(self)

        def _solve(self, from_memory=None) -> None:
            if from_memory is None:
                from_memory = self._domain_factory().get_initial_state_distribution()
            self._solve_from(from_memory)

        def _solve_from(self, initial_belief: Distribution[D.T_state]) -> None:
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

        def _get_next_action_from_belief(
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
            else:
                return action

        def get_utility_from_belief(
            self, belief: Distribution[D.T_state]
        ) -> Value[D.T_value]:
            """Get the best value for an explicit belief state."""
            return self._solver.get_utility_from_belief(belief)

        def _is_solution_defined_for_from_belief(
            self, belief: Distribution[D.T_state]
        ) -> bool:
            """Check if a solution is defined for an explicit belief state."""
            return self._solver.is_solution_defined_for_from_belief(belief)

        def reset_belief(self) -> None:
            """Reset the tracked belief to the initial belief from solve()."""
            self._solver.reset_belief()

        def get_nb_explored_beliefs(self) -> int:
            """Get the number of belief nodes in the search graph."""
            return self._solver.get_nb_explored_beliefs()

        def get_explored_beliefs(
            self,
        ) -> list[Distribution[D.T_state]]:
            """Get the set of all explored belief nodes, each as a
            distribution over states."""
            return self._solver.get_explored_beliefs()

        def get_nb_rollouts(self) -> int:
            """Get the number of trials performed."""
            return self._solver.get_nb_rollouts()

        def get_solving_time(self) -> int:
            """Get the solving time in milliseconds."""
            return self._solver.get_solving_time()

        def get_last_trajectory(
            self,
        ) -> list[tuple[dict[D.T_state, float], D.T_agent[D.T_concurrency[D.T_event]]]]:
            """Get the ordered list of (belief, action) pairs visited during the last RTDP-Bel trial.

            Returns the trajectory (path) explored during the most recent trial from
            the root belief. Each element is a tuple of (belief, action) where the belief
            is represented as a dictionary mapping states to probabilities, and the action
            is the best action selected in that belief during the trial. The trajectory
            begins with the root belief and ends at the deepest belief reached before the
            trial terminated (due to goal, depth limit, or time limit).

            The last element's action is the action selected in the final belief (or a
            default-constructed action if the final belief is a goal).

            This is useful for:
            - Replaying trajectories from the initial belief
            - Debugging algorithm behavior (which beliefs and actions were explored?)
            - Visualizing/logging the search process through belief space
            - Analyzing convergence patterns in the callback

            Returns an empty list if solve() has not been called yet.
            """
            return self._solver.get_last_trajectory()

        def get_belief_policy(
            self,
        ) -> dict[
            frozenset[tuple[D.T_state, float]],
            tuple[D.T_agent[D.T_concurrency[D.T_event]], Value[D.T_value]],
        ]:
            """Get the full belief-space policy as a dictionary.

            Keys are frozensets of (state, probability) tuples representing
            discretized beliefs. Values are (action, value) tuples.
            """
            return self._solver.get_belief_policy()

except ImportError:
    print(
        'Scikit-decide C++ hub library not found. Please check it is installed in "skdecide/hub".'
    )
    raise
