# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import sys
from typing import Any, Callable, List, Set, Tuple

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
from skdecide.hub.space.gym import ListSpace

record_sys_path = sys.path
skdecide_cpp_extension_lib_path = os.path.abspath(hub.__path__[0])
if skdecide_cpp_extension_lib_path not in sys.path:
    sys.path.append(skdecide_cpp_extension_lib_path)

try:

    from __skdecide_hub_cpp import _IWSolver_ as iw_solver

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

    class IW(ParallelSolver, Solver, DeterministicPolicies, Utilities, FromAnyState):
        """This is the skdecide implementation of the Iterated Width algorithm as described
        in "Width and Serialization of Classical Planning Problems" by Nir Lipovetzky
        and Hector Geffner (2012)
        """

        T_domain = D

        def __init__(
            self,
            domain_factory: Callable[[], Domain],
            state_features: Callable[[Domain, D.T_state], Any],
            use_state_feature_hash: bool = False,
            node_ordering: Callable[[float, int, int, float, int, int], bool] = (
                lambda a_gscore, a_novelty, a_depth, b_gscore, b_novelty, b_depth: a_gscore
                > b_gscore
            ),
            time_budget: int = 0,
            parallel: bool = False,
            shared_memory_proxy=None,
            callback: Callable[[IW], bool] = lambda slv: False,
            verbose: bool = False,
        ) -> None:
            """Construct a IW solver instance

            # Parameters
            domain_factory (Callable[[], Domain]): The lambda function to create a domain instance.
            state_features (Callable[[Domain, D.T_state], Any]): State feature vector
                used to compute the novelty measure
            use_state_feature_hash (bool, optional): Boolean indicating whether states
                must be hashed by using their features (True) or by using their native
                hash function (False). Defaults to False.
            node_ordering (_type_, optional): Lambda function called to rank two search nodes
                A and B, taking as inputs A's g-score, A's novelty, A's search depth,
                B's g-score, B's novelty, B's search depth, and returning true when B should be
                preferred to A (defaults to rank nodes based on their g-scores).
                Defaults to ( lambda a_gscore, a_novelty, a_depth, b_gscore, b_novelty, b_depth: a_gscore > b_gscore ).
            time_budget (int, optional): Maximum time allowed (in milliseconds) to continue searching
                for better plans after a first plan reaching a goal has been found. Defaults to 0.
            parallel (bool, optional): Parallelize the generation of state-action transitions
                on different processes using duplicated domains (True) or not (False). Defaults to False.
            shared_memory_proxy (_type_, optional): The optional shared memory proxy. Defaults to None.
            callback (_type_, optional): Lambda function called before popping
                the next state from the (priority) open queue, taking as arguments the solver and the domain,
                and returning true if the solver must be stopped. Defaults to (lambda slv:False).
            verbose (bool, optional): Boolean indicating whether verbose messages should be
                logged (True) or not (False). Defaults to False.
            """
            Solver.__init__(self, domain_factory=domain_factory)
            ParallelSolver.__init__(
                self,
                parallel=parallel,
                shared_memory_proxy=shared_memory_proxy,
            )
            self._lambdas = [state_features]
            self._ipc_notify = True

            self._solver = iw_solver(
                solver=self,
                domain=self.get_domain(),
                state_features=(
                    (lambda d, s: state_features(d, s))
                    if not self._parallel
                    else (lambda d, s: d.call(None, 0, s))
                ),
                use_state_feature_hash=use_state_feature_hash,
                node_ordering=node_ordering,
                time_budget=time_budget,
                parallel=parallel,
                callback=callback,
                verbose=verbose,
            )

        def close(self):
            """Joins the parallel domains' processes.
            Not calling this method (or not using the 'with' context statement)
            results in the solver forever waiting for the domain processes to exit.
            """
            if self._parallel:
                self._solver.close()
            ParallelSolver.close(self)

        def _reset(self) -> None:
            """Clears the search graph."""
            self._solver.clear()

        def _solve_from(self, memory: D.T_memory[D.T_state]) -> None:
            """Run the IW algorithm from a given root solving state

            # Parameters
            memory (D.T_memory[D.T_state]): State from which IW graph traversals
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
                which is why it is advised to call `IW.is_solution_defined_for` before

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
                it is advised to call `IW.is_solution_defined_for` before

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

        def get_nb_of_pruned_states(self) -> int:
            """Get the number of states pruned by the novelty measure among the
                ones present in the search graph

            # Returns
            int: Number of states pruned by the novelty
                measure among the ones present in the search graph graph
            """
            return self._solver.get_nb_of_pruned_states()

        def get_nb_tip_states(self) -> int:
            """Get the number of states present in the priority queue (i.e. those
                explored states that have not been yet closed by IW) of the current width
                search procedure (throws a runtime exception if no active width sub-solver
                is active)

            !!! warning
                Throws a runtime exception if no active width sub-solver is active

            # Returns
            int: Number of states present in the (priority) open queue
                of the current width search procedure
            """
            return self._solver.get_nb_tip_states()

        def get_top_tip_state(self) -> D.T_agent[D.T_observation]:
            """Get the top tip state, i.e. the tip state with the lowest
                lexicographical score (according to the node ordering functor given in the
                IWSolver instance's constructor) of the current width search procedure

            !!! warning
                Returns None if no active width sub-solver is active or if the priority queue
                of the current width search procedure is empty

            # Returns
            D.T_agent[D.T_observation]: Next tip state to be closed by the current width
                search procedure
            """
            return self._solver.get_top_tip_state()

        def get_intermediate_scores(self) -> List[Tuple[int, int, float]]:
            """Get the history of tuples of time point (in milliseconds), current
                width, and root state's f-score, recorded each time a goal state is
                encountered during the search

            # Returns
            List[Tuple[int, int, float]]: List of tuples of time point (in milliseconds),
                current width, and root state's f-score
            """
            return self._solver.get_intermediate_scores()

except ImportError:
    sys.path = record_sys_path
    print(
        'Scikit-decide C++ hub library not found. Please check it is installed in "skdecide/hub".'
    )
    raise
