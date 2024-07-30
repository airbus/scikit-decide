# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import sys
from typing import Callable, Dict, Optional, Tuple

from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    FloatHyperparameter,
    IntegerHyperparameter,
)

from skdecide import Domain, Solver, hub
from skdecide.builders.domain import (
    Actions,
    FullyObservable,
    Goals,
    Markovian,
    MultiAgent,
    PositiveCosts,
    Sequential,
    Simulation,
)
from skdecide.builders.solver import DeterministicPolicies, FromAnyState, Utilities
from skdecide.core import Value

record_sys_path = sys.path
skdecide_cpp_extension_lib_path = os.path.abspath(hub.__path__[0])
if skdecide_cpp_extension_lib_path not in sys.path:
    sys.path.append(skdecide_cpp_extension_lib_path)

try:

    from __skdecide_hub_cpp import _MARTDPSolver_ as martdp_solver

    # TODO: remove Markovian req?
    class D(
        Domain,
        MultiAgent,
        Sequential,
        Simulation,
        Actions,
        Goals,
        Markovian,
        FullyObservable,
        PositiveCosts,
    ):
        pass

    class MARTDP(Solver, DeterministicPolicies, Utilities, FromAnyState):
        """This is an experimental implementation of a skdecide-specific
        centralized multi-agent version of the RTDP algorithm ("Learning to act
        using real-time dynamic programming" by Barto, Bradtke and Singh, AIJ 1995)
        where the team's cost is the sum of individual costs and the joint applicable
        actions in a given joint state are sampled to avoid a combinatorial explosion
        of the joint action branching factor. This algorithm can (currently) only run
        on a single CPU.
        """

        T_domain = D

        hyperparameters = [
            IntegerHyperparameter(name="rollout_budget", low=10, high=1000000),
            IntegerHyperparameter(name="max_depth"),
            IntegerHyperparameter(name="max_feasibility_trials"),
            FloatHyperparameter(name="graph_expansion_rate"),
            IntegerHyperparameter(name="residual_moving_average_window"),
            FloatHyperparameter(name="epsilon"),
            FloatHyperparameter(name="discount"),
            FloatHyperparameter(name="action_choice_noise"),
            FloatHyperparameter(name="dead_end_cost"),
            CategoricalHyperparameter(
                name="continuous_planning", choices=[True, False]
            ),
        ]

        def __init__(
            self,
            domain_factory: Callable[[], T_domain],
            heuristic: Callable[
                [T_domain, D.T_state],
                Tuple[
                    D.T_agent[Value[D.T_value]],
                    D.T_agent[D.T_concurrency[D.T_event]],
                ],
            ] = lambda d, s: (
                {a: Value(cost=0) for a in s},
                {a: None for a in s},
            ),
            time_budget: int = 3600000,
            rollout_budget: int = 100000,
            max_depth: int = 1000,
            max_feasibility_trials: int = 0,  # will then choose nb_agents if 0
            graph_expansion_rate: float = 0.1,
            residual_moving_average_window: int = 100,
            epsilon: float = 0.0,  # not a stopping criterion by default
            discount: float = 1.0,
            action_choice_noise: float = 0.1,
            dead_end_cost: float = 10000,
            online_node_garbage: bool = False,
            continuous_planning: bool = True,
            callback: Callable[[MARTDP], bool] = lambda slv: False,
            verbose: bool = False,
        ) -> None:
            """Construct a MA-RTDP solver instance

            # Parameters
            domain_factory (Callable[[], T_domain], optional): The lambda function to create a domain instance.
            heuristic (Callable[ [T_domain, D.T_state], Tuple[ D.T_agent[Value[D.T_value]], D.T_agent[D.T_concurrency[D.T_event]], ], ], optional):
                Lambda function taking as arguments the domain and a state, and returning a pair of
                dictionary from agents to the individual heuristic estimates from the state to the goal,
                and of dictionary from agents to best guess individual actions.
                Defaults to lambda d, s: ({a: Value(cost=0) for a in s}, {a: None for a in s},).
            time_budget (int, optional): Maximum solving time in milliseconds. Defaults to 3600000.
            rollout_budget (int, optional): Maximum number of rollouts. Defaults to 100000.
            max_depth (int, optional): Maximum depth of each MA-RTDP trial (rollout). Defaults to 1000.
            max_feasibility_trials (int, optional): Number of trials for a given agent's applicable action
                to insert it in the joint applicable action set by reshuffling the agents' actions
                applicability ordering (set to the number of agents in the domain if it is equal to 0
                in this constructor). Defaults to 0.
            residual_moving_average_window (int, optional): Number of latest computed residual values
                to memorize in order to compute the average Bellman error (residual) at the root state
                of the search. Defaults to 100.
            epsilon (float, optional): Maximum Bellman error (residual) allowed to decide that a state
                is solved, or to decide when no labels are used that the value function of the root state
                of the search has converged (in the latter case: the root state's Bellman error is averaged
                over the residual_moving_average_window). Defaults to 0.001.
            discount (float, optional): Value function's discount factor. Defaults to 1.0.
            action_choice_noise (float, optional): Bernoulli probability of choosing an agent's
                random applicable action instead of the best current one when trying to
                generate a feasible joint applicable action from another agent's viewpoint. Defaults to 0.1.
            dead_end_cost (float, optional): Cost of a joint dead-end state (note that the
                transition cost function which is independently decomposed over the agents
                cannot easily model such joint dead-end state costs, which is why we allow
                for setting this global dead-end cost in this constructor). Defaults to 10000.
            online_node_garbage (bool, optional): Boolean indicating whether the search graph which is
                no more reachable from the root solving state should be deleted (True) or not (False). Defaults to False.
            continuous_planning (bool, optional): Boolean whether the solver should optimize again the policy
                from the current solving state (True) or not (False) even if the policy is already defined
                in this state. Defaults to True.
            callback (Callable[[MARTDP], bool], optional): Function called at the end of each MA-RTDP trial,
                taking as arguments the solver, and returning True if the solver must be stopped.
                The `MARTDP.get_domain` method callable on the solver instance can be used to retrieve
                the user domain. Defaults to (lambda slv: False).
            verbose (bool, optional): Boolean indicating whether verbose messages should be logged (True)
                or not (False). Defaults to False.
            """

            Solver.__init__(self, domain_factory=domain_factory)
            self._domain = self._domain_factory()
            self._continuous_planning = continuous_planning
            self._ipc_notify = True

            self._solver = martdp_solver(
                solver=self,
                domain=self.get_domain(),
                goal_checker=lambda d, s: d.is_goal(s),
                heuristic=lambda d, s: heuristic(d, s),
                time_budget=time_budget,
                rollout_budget=rollout_budget,
                max_depth=max_depth,
                max_feasibility_trials=max_feasibility_trials,
                graph_expansion_rate=graph_expansion_rate,
                residual_moving_average_window=residual_moving_average_window,
                epsilon=epsilon,
                discount=discount,
                action_choice_noise=action_choice_noise,
                dead_end_cost=dead_end_cost,
                online_node_garbage=online_node_garbage,
                callback=callback,
                verbose=verbose,
            )

        def _reset(self) -> None:
            """Clears the search graph."""
            self._solver.clear()

        def _solve_from(self, memory: D.T_memory[D.T_state]) -> None:
            """Run the MA-RTDP algorithm from a given root solving joint state

            # Parameters
            memory (D.T_memory[D.T_state]): Joint state from which to run the MA-RTDP
                algorithm (root of the search graph)
            """
            self._solver.solve(memory)

        def _is_solution_defined_for(
            self, observation: D.T_agent[D.T_observation]
        ) -> bool:
            """Indicates whether the solution policy is defined for a given joint state

            # Parameters
            observation (D.T_agent[D.T_observation]): Joint state for which an entry is
                searched in the policy graph

            # Returns
            bool: True if the state has been explored and an action is defined in this state,
                False otherwise
            """
            return self._solver.is_solution_defined_for(observation)

        def _get_next_action(
            self, observation: D.T_agent[D.T_observation]
        ) -> D.T_agent[D.T_concurrency[D.T_event]]:
            """Get the best computed joint action in terms of best Q-value in a given joint state.
                The search subgraph which is no more reachable after executing the returned action is
                also deleted if node garbage was set to True in the MA-RTDP instance's constructor.
                The solver is run from `observation` if `continuous_planning` was set to True
                in the MA-RTDP instance's constructor or if no solution is defined (i.e. has been
                previously computed) in `observation`.

            !!! warning
                Returns a random action if no action is defined in the given state,
                which is why it is advised to call `MARTDP.is_solution_defined_for` before

            # Parameters
            observation (D.T_agent[D.T_observation]): Joint state for which the best action
                is requested

            # Returns
            D.T_agent[D.T_concurrency[D.T_event]]: Best computed joint action
            """
            if self._continuous_planning or not self._is_solution_defined_for(
                observation
            ):
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

        def _get_utility(
            self, observation: D.T_agent[D.T_observation]
        ) -> D.T_agent[Value[D.T_value]]:
            """Get the best Q-value in a given joint state

            !!! warning
                Returns None if no action is defined in the given state, which is why
                it is advised to call `MARTDP.is_solution_defined_for` before

            # Parameters
            observation (D.T_agent[D.T_observation]): Joint state from which the best Q-value
                    is requested

            # Returns
            D.T_agent[Value[D.T_value]]: Maximum Q-value of the given joint state over the
                applicable joint actions in this state
            """
            return self._solver.get_utility(observation)

        def get_domain(self) -> T_domain:
            """Get the domain used by the MARTDP solver

            # Returns
                T_domain: Domain instance created by the MARTDP solver
            """
            return self._domain

        def get_nb_explored_states(self) -> int:
            """Get the number of states present in the search graph (which can be
                lower than the number of actually explored states if node garbage was
                set to True in the MA-RTDP instance's constructor)

            # Returns
            int: Number of states present in the search graph
            """
            return self._solver.get_nb_explored_states()

        def get_nb_rollouts(self) -> int:
            """Get the number of rollouts since the beginning of the search from
                the root solving state

            # Returns
            int: Number of rollouts (MA-RTDP trials)
            """
            return self._solver.get_nb_rollouts()

        def get_state_nb_actions(self, observation: D.T_agent[D.T_observation]) -> int:
            """Get the number of joint applicable actions generated so far in the
                given joint state (throws a runtime error exception if the given state is
                not present in the search graph, which can happen for instance when node
                garbage is set to true in the MARTDP instance's constructor and the
                non-reachable part of the search graph has been erased when calling the
                MARTDPSolver::get_best_action method)

            # Parameters
            observation (D.T_agent[D.T_observation]): Joint state from which the
                number of generated applicable actions is requested

            # Returns
            int: Number of generated applicable joint actions in the given state
            """
            return self._solver.get_state_nb_actions(observation)

        def get_residual_moving_average(self) -> float:
            """Get the average Bellman error (residual) at the root state of the search,
                or an infinite value if the number of computed residuals is lower than
                the epsilon moving average window set in the MARTDP instance's constructor

            # Returns
            float: Bellman error at the root state of the search averaged over
                the epsilon moving average window
            """
            return self._solver.get_residual_moving_average()

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
            Tuple[D.T_agent[D.T_concurrency[D.T_event]], D.T_agent[Value[D.T_value]]],
        ]:
            """Get the (partial) solution policy defined for the states for which
                the Q-value has been updated at least once (which is optimal if the
                algorithm has converged and labels are used)

            !!! warning
                Only defined over the states reachable from the last root solving state
                when node garbage was set to True in the MA-RTDP instance's constructor

            # Returns
            Dict[ D.T_agent[D.T_observation], Tuple[D.T_agent[D.T_concurrency[D.T_event]], D.T_agent[Value[D.T_value]]], ]:
                Mapping from joint states to pairs of joint action and best Q-value
            """
            return self._solver.get_policy()

except ImportError:
    sys.path = record_sys_path
    print(
        'Scikit-decide C++ hub library not found. Please check it is installed in "skdecide/hub".'
    )
    raise
