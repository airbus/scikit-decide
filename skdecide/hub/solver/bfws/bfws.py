# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import sys
from typing import Any, Callable, Dict, List, Set, Tuple

from skdecide import Domain, Solver, hub
from skdecide.builders.domain import (
    Actions,
    DeterministicInitialized,
    DeterministicTransitions,
    FullyObservable,
    Markovian,
    Rewards,
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

    from __skdecide_hub_cpp import _BFWSSolver_ as bfws_solver

    class D(
        Domain,
        SingleAgent,
        Sequential,
        DeterministicTransitions,
        Actions,
        DeterministicInitialized,
        Markovian,
        FullyObservable,
        Rewards,
    ):  # TODO: check why DeterministicInitialized & PositiveCosts/Rewards?
        pass

    class BFWS(ParallelSolver, Solver, DeterministicPolicies, Utilities, FromAnyState):
        """This is the skdecide implementation Best First Width Search from
        "Best-First Width Search: Exploration and Exploitation in Classical Planning"
        by Nir Lipovetzky and Hector Geffner (2017)
        """

        T_domain = D

        def __init__(
            self,
            domain_factory: Callable[[], Domain],
            state_features: Callable[[Domain, D.T_state], Any],
            heuristic: Callable[
                [Domain, D.T_state], D.T_agent[Value[D.T_value]]
            ] = lambda d, s: Value(cost=0),
            parallel: bool = False,
            shared_memory_proxy=None,
            callback: Callable[[BFWS], bool] = lambda slv: False,
            verbose: bool = False,
        ) -> None:
            """Construct a BFWS solver instance

            # Parameters
            domain_factory (Callable[[], Domain]): The lambda function to create a domain instance.
            state_features (Callable[[Domain, D.T_state], Any]): State feature vector
                used to compute the novelty measure
            heuristic (Callable[[Domain, D.T_state], D.T_agent[Value[D.T_value]]]):
                Lambda function taking as arguments the domain and a state object,
                and returning the heuristic estimate from the state to the goal.
                Defaults to (lambda d, s: Value(cost=0)).
            parallel (bool, optional): Parallelize the generation of state-action transitions
                on different processes using duplicated domains (True) or not (False). Defaults to False.
            shared_memory_proxy (_type_, optional): The optional shared memory proxy. Defaults to None.
            callback (Callable[[BFWS], bool], optional): Lambda function called before popping
                the next state from the (priority) open queue, taking as arguments the solver and the domain,
                and returning true if the solver must be stopped. Defaults to (lambda slv: False).
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
            self._domain = None
            self._state_features = state_features
            self._heuristic = heuristic
            self._lambdas = [
                self._state_features,
                self._heuristic,
            ]
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
            self._solver = bfws_solver(
                solver=self,
                domain=self.get_domain(),
                goal_checker=lambda d, s: d.is_goal(s),
                state_features=(
                    (lambda d, s: self._state_features(d, s))
                    if not self._parallel
                    else (lambda d, s: d.call(None, 0, s))
                ),
                heuristic=(
                    (lambda d, s: self._heuristic(d, s))
                    if not self._parallel
                    else (lambda d, s: d.call(None, 1, s))
                ),
                parallel=self._parallel,
                callback=self._callback,
                verbose=self._verbose,
            )
            self._solver.clear()

        def _reset(self) -> None:
            """Clears the search graph."""
            self._solver.clear()

        def _solve_from(self, memory: D.T_memory[D.T_state]) -> None:
            """Run the BFWS algorithm from a given root solving state

            # Parameters
            memory (D.T_memory[D.T_state]): State from which BFWS graph traversals
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
            """Get the best computed action in terms of minimum cost-to-go in a given state.
                The solver is run from `observation` if no solution is defined (i.e. has been
                previously computed) in `observation`.

            !!! warning
                Returns a random action if no action is defined in the given state,
                which is why it is advised to call `BFWS.is_solution_defined_for` before

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
            """Get the minimum cost-to-go in a given state

            !!! warning
                Returns None if no action is defined in the given state, which is why
                it is advised to call `BFWS.is_solution_defined_for` before

            # Parameters
            observation (D.T_agent[D.T_observation]): State from which the minimum cost-to-go is requested

            # Returns
            D.T_value: Minimum cost-to-go of the given state over the applicable actions in this state
            """
            return self._solver.get_utility(observation)

        def get_nb_explored_states(self) -> int:
            """Get the number of states present in the search graph

            # Returns
            int: Number of states present in the search graph
            """
            return self._solver.get_nb_explored_states()

        def get_explored_states(self) -> Set[D.T_agent[D.T_observation]]:
            """Get the set of states present in the search graph (i.e. the graph's
                state nodes minus the nodes' encapsulation and their neighbors)

            # Returns
            Set[D.T_agent[D.T_observation]]: Set of states present in the search graph
            """
            return self._solver.get_explored_states()

        def get_nb_tip_states(self) -> int:
            """Get the number of states present in the priority queue (i.e. those
                explored states that have not been yet closed by BFWS)

            # Returns
            int: Number of states present in the (priority) open queue
            """
            return self._solver.get_nb_tip_states()

        def get_top_tip_state(self) -> D.T_agent[D.T_observation]:
            """Get the top tip state, i.e. the tip state with the lowest f-score

            !!! warning
                Returns None if the priority queue is empty

            # Returns
            D.T_agent[D.T_observation]: Next tip state to be closed by BFWS
            """
            return self._solver.get_top_tip_state()

        def get_solving_time(self) -> int:
            """Get the solving time in milliseconds since the beginning of the
                search from the root solving state

            # Returns
            int: Solving time in milliseconds
            """
            return self._solver.get_solving_time()

        def get_plan(
            self, observation: D.T_agent[D.T_observation]
        ) -> List[
            Tuple[
                D.T_agent[D.T_observation],
                D.T_agent[D.T_concurrency[D.T_event]],
                D.T_value,
            ]
        ]:
            """Get the solution plan starting in a given state

            !!! warning
                Returns an empty list if no plan has been previously computed that goes
                through the given state.
                Throws a runtime exception if a state cycle is detected in the plan

            # Parameters
            observation (D.T_agent[D.T_observation]): State from which a solution plan
                to a goal state is requested

            # Returns
            List[ Tuple[ D.T_agent[D.T_observation], D.T_agent[D.T_concurrency[D.T_event]], D.T_value, ] ]:
                Sequence of tuples of state, action and transition cost (computed as the
                difference of g-scores between this state and the next one) visited
                along the execution of the plan
            """
            return self._solver.get_plan(observation)

        def get_policy(
            self,
        ) -> Dict[
            D.T_agent[D.T_observation],
            Tuple[D.T_agent[D.T_concurrency[D.T_event]], D.T_value],
        ]:
            """Get the (partial) solution policy defined for the states for which
                a solution plan that goes through them has been previously computed at
                least once

            !!! warning
                Only defined over the states reachable from the root solving state

            # Returns
            Dict[ D.T_agent[D.T_observation], Tuple[D.T_agent[D.T_concurrency[D.T_event]], D.T_value], ]:
                Mapping from states to pairs of action and minimum cost-to-go
            """
            return self._solver.get_policy()

except ImportError:
    sys.path = record_sys_path
    print(
        'Scikit-decide C++ hub library not found. Please check it is installed in "skdecide/hub".'
    )
    raise
