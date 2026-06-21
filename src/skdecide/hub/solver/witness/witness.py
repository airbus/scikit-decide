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
    from skdecide.hub.__skdecide_hub_cpp import _WitnessSolver_ as witness_solver

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

    class Witness(
        ParallelSolver, Solver, DeterministicPolicies, Utilities, FromAnyState
    ):
        """Exact POMDP solver using the Witness algorithm.

        From: Littman, "The Witness Algorithm: Solving Partially Observable
        Markov Decision Processes", Brown University TR, 1994.

        Performs exact value iteration with piecewise-linear convex value
        functions represented as sets of alpha-vectors. Uses LP-based
        witness point finding to discover all non-dominated alpha-vectors
        and Monahan LP pruning to remove dominated ones.

        Intended for verifying correctness of approximate POMDP solvers
        on tiny test problems. Not suitable for large state/action spaces.

        The default interface works with observations. The solver internally
        maintains and updates the current belief using Bayes rule from the
        observation history.
        """

        T_domain = D

        class AlphaVectorDict(TypedDict):
            """Type for alpha vector dictionaries returned by get_alpha_vectors().

            Fields:
            - values: dict mapping states (D.T_state) to Value[D.T_value]
            - action: action object (D.T_agent[D.T_concurrency[D.T_event]])
            """

            values: dict[D.T_state, Value[D.T_value]]
            action: D.T_agent[D.T_concurrency[D.T_event]]

        def __init__(
            self,
            domain_factory: Callable[[], Domain],
            epsilon: float = 0.001,
            discount: float = 0.95,
            max_iterations: int = 100,
            lp_infinity: float = 1e20,
            lp_tolerance: float = 1e-10,
            terminal_value: Callable[[object], Value] = lambda s: Value(reward=0),
            parallel: bool = False,
            shared_memory_proxy=None,
            callback: Callable[[Witness], bool] = lambda slv: False,
            verbose: bool = False,
        ) -> None:
            """Construct a Witness solver instance.

            # Parameters
            domain_factory: Lambda function to create a domain instance.
            epsilon: Convergence threshold for value function change at
                corner beliefs. Defaults to 0.001.
            discount: Discount factor gamma. Must be in (0, 1).
                Defaults to 0.95.
            max_iterations: Maximum number of value iteration steps.
                Defaults to 100.
            lp_infinity: Upper bound used for LP variable bounds with
                HiGHS. Defaults to 1e20.
            lp_tolerance: Numerical tolerance for LP feasibility checks
                and alpha-vector comparisons. Defaults to 1e-10.
            terminal_value: Function (state) -> Value returning the
                value for terminal non-goal states.
                Defaults to lambda s: Value(reward=0).
            parallel: Parallelize domain calls. Defaults to False.
            shared_memory_proxy: Optional shared memory proxy.
                Defaults to None.
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

            self._solver = witness_solver(
                solver=self,
                domain=self.get_domain(),
                epsilon=epsilon,
                discount=discount,
                max_iterations=max_iterations,
                lp_infinity=lp_infinity,
                lp_tolerance=lp_tolerance,
                terminal_value=lambda s: terminal_value(s),
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
            """Run Witness from an initial belief distribution.

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

        def get_utility_from_belief(
            self, belief: Distribution[D.T_state]
        ) -> Value[D.T_value]:
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
            """Get the number of alpha-vectors in the value function."""
            return self._solver.get_nb_alpha_vectors()

        def get_nb_iterations(self) -> int:
            """Get the number of value iteration steps performed."""
            return self._solver.get_nb_iterations()

        def get_solving_time(self) -> int:
            """Get the solving time in milliseconds."""
            return self._solver.get_solving_time()

        def get_callback_event(self) -> str:
            return self._solver.get_callback_event()

        def get_alpha_vectors(self) -> list[AlphaVectorDict]:
            """Get the alpha-vectors representing the policy.

            Returns a list of dictionaries, each containing:
            - 'values': dict[D.T_state, Value[D.T_value]] - state to Value mapping
            - 'action': D.T_agent[D.T_concurrency[D.T_event]] - associated action

            The policy at any belief b is: choose the action of the alpha-vector
            that maximizes sum(b[s] * alpha['values'][s].reward for all states s).
            """
            return self._solver.get_alpha_vectors()

except ImportError:
    print(
        "Scikit-decide C++ hub library not found. "
        "Witness solver requires the C++ hub with HiGHS support."
    )
    raise
