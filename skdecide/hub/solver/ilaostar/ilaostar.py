# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import sys
from typing import Callable, Dict, Optional, Tuple

from skdecide import Domain, Solver, hub
from skdecide.builders.domain import (
    Actions,
    EnumerableTransitions,
    FullyObservable,
    Goals,
    Markovian,
    PositiveCosts,
    Sequential,
    SingleAgent,
)
from skdecide.builders.solver import (
    DeterministicPolicies,
    FromAnyState,
    ParallelSolver,
    Utilities,
)
from skdecide.core import Value

record_sys_path = sys.path
skdecide_cpp_extension_lib_path = os.path.abspath(hub.__path__[0])
if skdecide_cpp_extension_lib_path not in sys.path:
    sys.path.append(skdecide_cpp_extension_lib_path)

try:

    from __skdecide_hub_cpp import _ILAOStarSolver_ as ilaostar_solver

    # TODO: remove Markovian req?
    class D(
        Domain,
        SingleAgent,
        Sequential,
        EnumerableTransitions,
        Actions,
        Goals,
        Markovian,
        FullyObservable,
        PositiveCosts,
    ):
        pass

    class ILAOstar(
        ParallelSolver, Solver, DeterministicPolicies, Utilities, FromAnyState
    ):
        """This is the skdecide implementation of Improved-LAO* as described in
        "LAO*: A heuristic search algorithm that finds solutions with loops" by Eric
        A. Hansen and Shlomo Zilberstein (2001)
        """

        T_domain = D

        def __init__(
            self,
            domain_factory: Callable[[], Domain],
            heuristic: Optional[
                Callable[[Domain, D.T_state], D.T_agent[Value[D.T_value]]]
            ] = lambda d, s: Value(cost=0),
            discount: float = 1.0,
            epsilon: float = 0.001,
            parallel: bool = False,
            shared_memory_proxy=None,
            callback: Callable[[ILAOstar], bool] = lambda slv: False,
            verbose: bool = False,
        ) -> None:
            """Construct a ILAO* solver instance

            # Parameters
            domain_factory (Callable[[], Domain]): The lambda function to create a domain instance.
            heuristic (Optional[ Callable[[Domain, D.T_state], D.T_agent[Value[D.T_value]]] ], optional):
                Lambda function taking as arguments the domain and a state object,
                and returning the heuristic estimate from the state to the goal.
                Defaults to (lambda d, s: Value(cost=0)).
            discount (float, optional): Value function's discount factor. Defaults to 1.0.
            epsilon (float, optional): Maximum Bellman error (residual) allowed to decide that
                the value function of the root state of the search has converged. Defaults to 0.001.
            parallel (bool, optional): Parallelize the generation of state-action transitions and the update
                of state attributes (e.g. Bellman residuals) on different processes using duplicated domains (True)
                or not (False). Defaults to False.
            shared_memory_proxy (_type_, optional): The optional shared memory proxy. Defaults to None.
            callback (_type_, optional): Lambda function called at the beginning of each policy update
                depth-first search, taking as arguments the solver and the domain, and returning true if
                the solver must be stopped. Defaults to (lambda slv: False).
            verbose (bool, optional): Boolean indicating whether verbose messages should be
                logged (True) or not (False). Defaults to False.
            """
            ParallelSolver.__init__(
                self,
                parallel=parallel,
                shared_memory_proxy=shared_memory_proxy,
            )
            Solver.__init__(self, domain_factory=domain_factory)
            self._solver = None
            self._discount = discount
            self._epsilon = epsilon
            self._heuristic = heuristic
            self._lambdas = [self._heuristic]
            self._callback = callback
            self._verbose = verbose
            self._ipc_notify = True

        def close(self):
            """Joins the parallel domains' processes.
            Not calling this method (or not using the 'with' context statement)
            results in the solver forever waiting for the domain processes to exit.
            """
            if self._parallel:
                self._solver.close()
            ParallelSolver.close(self)

        def _init_solve(self) -> None:
            self._solver = ilaostar_solver(
                solver=self,
                domain=self.get_domain(),
                goal_checker=lambda d, s: d.is_goal(s),
                heuristic=lambda d, s: (
                    self._heuristic(d, s) if not self._parallel else d.call(None, 0, s)
                ),
                discount=self._discount,
                epsilon=self._epsilon,
                parallel=self._parallel,
                callback=self._callback,
                verbose=self._verbose,
            )
            self._solver.clear()

        def _reset(self) -> None:
            """Clears the search graph."""
            self._solver.clear()

        def _solve_from(self, memory: D.T_memory[D.T_state]) -> None:
            """Run the ILAO* algorithm from a given root solving state

            # Parameters
            memory (D.T_memory[D.T_state]): State from which ILAO* graph traversals
                are performed (root of the search graph)
            """
            self._solver.solve(memory)

        def _is_solution_defined_for(
            self, observation: D.T_agent[D.T_observation]
        ) -> bool:
            """Indicates whether the solution policy (potentially built from merging
                several previously computed plans) is defined for a given state

            # Parameters
            observation (D.T_agent[D.T_observation]): State for which an entry is searched
                in the policy graph

            # Returns
            bool: True if a plan that goes through the state has been previously computed,
                False otherwise
            """
            return self._solver.is_solution_defined_for(observation)

        def _get_next_action(
            self, observation: D.T_agent[D.T_observation]
        ) -> D.T_agent[D.T_concurrency[D.T_event]]:
            """Get the best computed action in terms of of best Q-value in a given state.
                The solver is run from `observation` if no solution is defined (i.e. has been
                previously computed) in `observation`.

            !!! warning
                Returns a random action if no action is defined in the given state,
                which is why it is advised to call `ILAOstar.is_solution_defined_for` before

            # Parameters
            observation (D.T_agent[D.T_observation]): State for which the best action is requested

            # Returns
            D.T_agent[D.T_concurrency[D.T_event]]: Best computed action
            """
            if not self._is_solution_defined_for(observation):
                self._solve_from(observation)
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
            else:
                return action

        def _get_utility(self, observation: D.T_agent[D.T_observation]) -> D.T_value:
            """Get the best Q-value in a given state

            !!! warning
                Returns None if no action is defined in the given state, which is why
                it is advised to call `ILAOstar.is_solution_defined_for` before

            # Parameters
            observation (D.T_agent[D.T_observation]): State from which the best Q-value is requested

            # Returns
            D.T_value: Minimum Q-Value of the given state over the applicable actions in this state
            """
            return self._solver.get_utility(observation)

        def get_nb_of_explored_states(self) -> int:
            """Get the number of states present in the search graph

            # Returns
            int: Number of states present in the search graph
            """
            return self._solver.get_nb_of_explored_states()

        def get_explored_states(self) -> Set[D.T_agent[D.T_observation]]:
            """Get the set of states present in the search graph (i.e. the graph's
                state nodes minus the nodes' encapsulation and their children)

            # Returns
            Set[D.T_agent[D.T_observation]]: Set of states present in the search graph
            """
            return self._solver.get_explored_states()

        def best_solution_graph_size(self) -> int:
            """Get the number of states present in the current best policy graph

            # Returns
            int: Number of states present in the current best policy graph
            """
            return self._solver.best_solution_graph_size()

        def get_solving_time(self) -> int:
            """Get the solving time in milliseconds since the beginning of the
                search from the root solving state

            # Returns
            int: Solving time in milliseconds
            """
            return self._solver.get_solving_time()

        def get_policy(
            self,
        ) -> Dict[
            D.T_agent[D.T_observation],
            Tuple[D.T_agent[D.T_concurrency[D.T_event]], float],
        ]:
            """Get the (partial) solution policy defined for the states for which
                the Q-value has been updated at least once (which is optimal for
                the states reachable by the policy from the root solving state)

            !!! warning
                Only defined over the states reachable from the root solving state

            # Returns
            Dict[ D.T_agent[D.T_observation], Tuple[D.T_agent[D.T_concurrency[D.T_event]], D.T_value], ]:
                Mapping from states to pairs of action and best Q-value
            """
            return self._solver.get_policy()

except ImportError:
    sys.path = record_sys_path
    print(
        'Scikit-decide C++ hub library not found. Please check it is installed in "skdecide/hub".'
    )
    raise
