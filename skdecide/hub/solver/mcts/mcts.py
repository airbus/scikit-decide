# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import random as rd
import sys
from enum import Enum
from math import sqrt
from typing import Callable, Dict, List, Optional, Tuple

from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    EnumHyperparameter,
    FloatHyperparameter,
    IntegerHyperparameter,
)

from skdecide import Domain, Solver, hub
from skdecide.builders.domain import (
    Actions,
    DeterministicInitialized,
    Environment,
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

    from __skdecide_hub_cpp import _MCTSOptions_ as mcts_options
    from __skdecide_hub_cpp import _MCTSSolver_ as mcts_solver

    class D(
        Domain,
        SingleAgent,
        Sequential,
        Environment,
        Actions,
        DeterministicInitialized,
        Markovian,
        FullyObservable,
        Rewards,
    ):  # TODO: check why DeterministicInitialized & PositiveCosts/Rewards?
        pass

    class MCTS(ParallelSolver, Solver, DeterministicPolicies, Utilities, FromAnyState):
        """This is the skdecide implementation of MCTS and UCT from
        "A Survey of Monte Carlo Tree Search Methods" by Browne et al
        (IEEE Transactions on Computational Intelligence  and AI in games,
        2012). We additionally implement a heuristic value estimate as in
        "Monte-Carlo tree search and rapid action value estimation in
        computer Go" by Gelly and Silver (Artificial Intelligence, 2011)
        except that the heuristic estimate is called on states but not
        on state-action pairs to be more in line with heuristic search
        algorithms in the literature and other implementations of
        heuristic search algorithms in skdecide.
        """

        T_domain = D

        class TransitionMode(Enum):
            STEP = mcts_options.TransitionMode.Step
            SAMPLE = mcts_options.TransitionMode.Sample
            DISTRIBUTION = mcts_options.TransitionMode.Distribution

        class TreePolicy(Enum):
            DEFAULT = mcts_options.TreePolicy.Default

        class Expander(Enum):
            FULL = mcts_options.Expander.Full
            PARTIAL = mcts_options.Expander.Partial

        class ActionSelector(Enum):
            UCB1 = mcts_options.ActionSelector.UCB1
            BEST_Q_VALUE = mcts_options.ActionSelector.BestQValue

        class RolloutPolicy(Enum):
            RANDOM = mcts_options.RolloutPolicy.Random
            CUSTOM = mcts_options.RolloutPolicy.Custom
            VOID = mcts_options.RolloutPolicy.Void

        class BackPropagator(Enum):
            GRAPH = mcts_options.BackPropagator.Graph

        hyperparameters = [
            IntegerHyperparameter(name="rollout_budget"),
            IntegerHyperparameter(name="max_depth"),
            IntegerHyperparameter(name="residual_moving_average_window"),
            FloatHyperparameter(name="epsilon"),
            FloatHyperparameter(name="discount"),
            FloatHyperparameter(name="ucb_constant"),
            FloatHyperparameter(name="state_expansion_rate"),
            FloatHyperparameter(name="action_expansion_rate"),
            EnumHyperparameter(
                name="transition_mode",
                enum=TransitionMode,
            ),
            EnumHyperparameter(name="tree_policy", enum=TreePolicy),
            EnumHyperparameter(name="expander", enum=Expander),
            EnumHyperparameter(
                name="action_selector_optimization",
                enum=ActionSelector,
            ),
            EnumHyperparameter(
                name="action_selector_execution",
                enum=ActionSelector,
            ),
            EnumHyperparameter(name="rollout_policy", enum=RolloutPolicy),
            EnumHyperparameter(
                name="back_propagator",
                enum=BackPropagator,
            ),
            CategoricalHyperparameter(
                name="continuous_planning", choices=[True, False]
            ),
        ]

        def __init__(
            self,
            domain_factory: Callable[[], T_domain],
            time_budget: int = 3600000,
            rollout_budget: int = 100000,
            max_depth: int = 1000,
            residual_moving_average_window: int = 100,
            epsilon: float = 0.0,  # not a stopping criterion by default
            discount: float = 1.0,
            ucb_constant: float = 1.0 / sqrt(2.0),
            online_node_garbage: bool = False,
            custom_policy: Optional[
                Callable[
                    [T_domain, D.T_agent[D.T_observation]],
                    D.T_agent[D.T_concurrency[D.T_event]],
                ]
            ] = None,
            heuristic: Optional[
                Callable[
                    [T_domain, D.T_agent[D.T_observation]],
                    Tuple[D.T_agent[Value[D.T_value]], int],
                ]
            ] = None,
            state_expansion_rate: float = 0.1,
            action_expansion_rate: float = 0.1,
            transition_mode: TransitionMode = TransitionMode.DISTRIBUTION,
            tree_policy: TreePolicy = TreePolicy.DEFAULT,
            expander: Expander = Expander.FULL,
            action_selector_optimization: ActionSelector = ActionSelector.UCB1,
            action_selector_execution: ActionSelector = ActionSelector.BEST_Q_VALUE,
            rollout_policy: RolloutPolicy = RolloutPolicy.RANDOM,
            back_propagator: BackPropagator = BackPropagator.GRAPH,
            continuous_planning: bool = True,
            parallel: bool = False,
            shared_memory_proxy=None,
            callback: Callable[[MCTS, Optional[int]], bool] = lambda slv, i=None: False,
            verbose: bool = False,
        ) -> None:
            """Construct a MCTS solver instance

            # Parameters
            domain_factory (Callable[[], T_domain]): The lambda function to create a domain instance.
            time_budget (int, optional): Maximum solving time in milliseconds. Defaults to 3600000.
            rollout_budget (int, optional): Maximum number of rollouts. Defaults to 100000.
            max_depth (int, optional): Maximum depth of each MCTS rollout. Defaults to 1000.
            residual_moving_average_window (int, optional): Number of latest computed residual values
                to memorize in order to compute the average Bellman error (residual) at the root state
                of the search. Defaults to 100.
            epsilon (float, optional): Maximum Bellman error (residual) allowed to decide that a state
                is solved, or to decide when no labels are used that the value function of the root state
                of the search has converged (in the latter case: the root state's Bellman error is averaged
                over the residual_moving_average_window). Defaults to 0.0.
            discount (float, optional): Value function's discount factor. Defaults to 1.0.
            ucb_constant (float, optional): UCB constant as used in the UCT algorithm when the action selector
                (for optimization or execution) is `MCTS.ActionSelector.UCB1`. Defaults to 1.0/sqrt(2.0).
            online_node_garbage (bool, optional): Boolean indicating whether the search graph which is
                no more reachable from the root solving state should be deleted (True) or not (False). Defaults to False.
            custom_policy (Callable[ [T_domain, D.T_agent[D.T_observation]], D.T_agent[D.T_concurrency[D.T_event]], ], optional):
                Custom policy function to use in the rollout policy from non-expanded state nodes when the rollout policy is
                `MCTS.RolloutPolicy.CUSTOM`. Defaults to None (no custom policy in use).
            heuristic (Callable[ [T_domain, D.T_agent[D.T_observation]], Tuple[D.T_agent[Value[D.T_value]], int], ], optional):
                Optional Heuristic function to initialize non-expanded state nodes (returns a pair of value estimate and
                fake number of visit counts). Defaults to None (no heuristic in use).
            state_expansion_rate (float, optional): Value $rs$ used when the expander is `MCTS.Expander.PARTIAL`
                such that the probability of discovering new applicable actions in a given state node with already $na$ discovered
                applicable actions is equal to $e^{-rs \cdot na}$. Defaults to 0.1.
            action_expansion_rate (float, optional):  Value $ra$ used when the expander is `MCTS.Expander.PARTIAL`
                such that the probability of discovering new state outcomes in a given action node with already $ns$ discovered
                state outcomes is equal to $e^{-ra \cdot ns}$. Defaults to 0.1.
            transition_mode (MCTS.TransitionMode, optional): Transition mode enum (one of `MCTS.TransitionMode.STEP`,
                `MCTS.TransitionMode.SAMPLE` or `MCTS.TransitionMode.DISTRIBUTION` to progress the
                trajectories with, respectively, the 'step' or 'sample' or 'get_next_state_distribution' method of the domain
                depending on the domain's dynamics capabilities). Defaults to `MCTS.TransitionMode.DISTRIBUTION`.
            tree_policy (MCTS.TreePolicy, optional): Tree policy enum (currently only
                `MCTS.TreePolicy.DEFAULT` which rollouts a random trajectory from the current root
                solving state until reaching a non-expanded state node of the tree). Defaults to `MCTS.TreePolicy.DEFAULT`.
            expander (MCTS.Expander, optional): Expander enum used when a state needs to be
                expanded (one of `MCTS.Expander.FULL` if applicable actions and next states should be
                all enumerated for each transition function, or `MCTS.Expander.PARTIAL` if they should
                be sampled with a probability which exponentially decreases as the number of already discovered
                applicable actions and next states increases). Defaults to `MCTS.Expander.FULL`.
            action_selector_optimization (MCTS.ActionSelector, optional): Action selector class used to
                select actions in the tree policy's trajectory simulations (one of
                `MCTS.ActionSelector.UCB1` to select the action based on the UCB criterion, or
                `MCTS.ActionSelector.BEST_Q_VALUE` to select the action with maximum Q-Value in the
                current state node). Defaults to `MCTS.ActionSelector.UCB1`.
            action_selector_execution (MCTS.ActionSelector, optional): Action selector class used to
                select actions at execution time when the 'get_best_action' method of the
                solver is invoked in a given execution state (one of
                `MCTS.ActionSelector.UCB1` to select the action based on the UCB criterion, or
                `MCTS.ActionSelector.BEST_Q_VALUE` to select the action with maximum Q-Value in the
                current state node). Defaults to `MCTS.ActionSelector.BEST_Q_VALUE`.
            rollout_policy (MCTS.RolloutPolicy, optional): Rollout policy enum (one of
                `MCTS.RolloutPolicy.RANDOM` to simulate trajectories starting in a non-expanded state
                node of the tree by sampling random applicable actions in each visited state, or
                `MCTS.RolloutPolicy.CUSTOM` to simulate them by applying actions from the given policy
                'custom_policy' given to this constructor, or `MCTS.RolloutPolicy.VOID` to deactivate
                the simulation of trajectories from non-expanded state nodes, in which latter case it is advised to
                provide the 'heuristic' function in this constructor to initialize non-expanded state nodes' values).
                Defaults to `MCTS.RolloutPolicy.RANDOM`.
            back_propagator (MCTS.BackPropagator, optional): Back propagator enum (currently only
                `MCTS.BackPropagator.GRAPH` which back-propagates empirical Q-values from non-expanded
                state nodes up to the root node of the tree along the tree policy's sampled
                trajectories). Defaults to `MCTS.BackPropagator.GRAPH`.
            continuous_planning (bool, optional): Boolean whether the solver should optimize again the policy
                from the current solving state (True) or not (False) even if the policy is already defined
                in this state. Defaults to True.
            parallel (bool, optional): Parallelize MCTS rollouts on different processes using duplicated domains (True)
                or not (False). Defaults to False.
            shared_memory_proxy (_type_, optional): The optional shared memory proxy. Defaults to None.
            callback (Callable[[MCTS, Optional[int]], optional): Function called at the end of each RIW rollout,
                taking as arguments the solver and the thread/process ID (i.e. parallel domain ID, which is equal to None
                in case of sequential execution, i.e. when 'parallel' is set to False in this constructor) from
                which the callback is called, and returning True if the solver must be stopped. The callback lambda
                function cannot take the (potentially parallelized) domain as argument because we could not otherwise
                serialize (i.e. pickle) the solver to pass it to the corresponding parallel domain process in case of parallel
                execution. Nevertheless, the `ParallelSolver.get_domain` method callable on the solver instance
                can be used to retrieve either the user domain in sequential execution, or the parallel domains proxy
                `ParallelDomain` in parallel execution from which domain methods can be called by using the
                callback's process ID argument. Defaults to (lambda slv, i=None: False).
            verbose (bool, optional): Boolean indicating whether verbose messages should be logged (True)
                or not (False). Defaults to False.
            """
            Solver.__init__(self, domain_factory=domain_factory)
            ParallelSolver.__init__(
                self,
                parallel=parallel,
                shared_memory_proxy=shared_memory_proxy,
            )
            self._solver = None
            self._domain = None
            self._continuous_planning = continuous_planning
            self._lambdas = [custom_policy, heuristic]
            self._ipc_notify = True

            self._solver = mcts_solver(
                solver=self,
                domain=self.get_domain(),
                time_budget=time_budget,
                rollout_budget=rollout_budget,
                max_depth=max_depth,
                residual_moving_average_window=residual_moving_average_window,
                epsilon=epsilon,
                discount=discount,
                ucb_constant=ucb_constant,
                online_node_garbage=online_node_garbage,
                custom_policy=(
                    None
                    if custom_policy is None
                    else (
                        (lambda d, s, i=None: custom_policy(d, s))
                        if not parallel
                        else (lambda d, s, i=None: d.call(i, 0, s))
                    )
                ),
                heuristic=(
                    None
                    if heuristic is None
                    else (
                        (lambda d, s, i=None: heuristic(d, s))
                        if not parallel
                        else (lambda d, s, i=None: d.call(i, 1, s))
                    )
                ),
                state_expansion_rate=state_expansion_rate,
                action_expansion_rate=action_expansion_rate,
                transition_mode=transition_mode.value,
                tree_policy=tree_policy.value,
                expander=expander.value,
                action_selector_optimization=action_selector_optimization.value,
                action_selector_execution=action_selector_execution.value,
                rollout_policy=rollout_policy.value,
                back_propagator=back_propagator.value,
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
            """Run the MCTS algorithm from a given root solving state

            # Parameters
            memory (D.T_memory[D.T_state]): Root state of the search grph from which
                MCTS rollouts are launched
            """
            self._solver.solve(memory)

        def _is_solution_defined_for(
            self, observation: D.T_agent[D.T_observation]
        ) -> bool:
            """Indicates whether the solution policy is defined for a given state

            # Parameters
            observation (D.T_agent[D.T_observation]): State for which an entry is searched
                in the policy graph

            # Returns
            bool: True if the state has been explored and an action can be obtained
                from the execution action selector, False otherwise
            """
            return self._solver.is_solution_defined_for(observation)

        def _get_next_action(
            self, observation: D.T_agent[D.T_observation]
        ) -> D.T_agent[D.T_concurrency[D.T_event]]:
            """Get the best action to execute in a given state according to the execution action selector.
                The search subgraph which is no more reachable after executing the returned action is
                also deleted if node garbage was set to True in the MCTS instance's constructor.
                The solver is run from `observation` if `continuous_planning` was set to True
                in the MCTS instance's constructor or if no solution is defined (i.e. has been
                previously computed) in `observation`.

            !!! warning
                Returns a random action if no action is defined in the given state,
                which is why it is advised to call `MCTS.is_solution_defined_for` before

            # Parameters
            observation (D.T_agent[D.T_observation]): State for which the best action is requested

            # Returns
            D.T_agent[D.T_concurrency[D.T_event]]: Best action to execute according to the
                execution action selector
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

        def _get_utility(self, observation: D.T_agent[D.T_observation]) -> D.T_value:
            """Get the best value in a given state according to the execution action selector

            !!! warning
                Returns None if no action is defined in the given state, which is why
                it is advised to call `MCTS.is_solution_defined_for` before

            # Parameters
            observation (D.T_agent[D.T_observation]): State from which the best value is requested

            # Returns
            D.T_value: Value of the action returned by the execution action selector
            """
            return self._solver.get_utility(observation)

        def get_nb_explored_states(self) -> int:
            """Get the number of states present in the search graph (which can be
                lower than the number of actually explored states if node garbage was
                set to True in the MCTS instance's constructor)

            # Returns
            int: Number of states present in the search graph
            """
            return self._solver.get_nb_explored_states()

        def get_nb_rollouts(self) -> int:
            """Get the number of rollouts since the beginning of the search from
                the root solving state

            # Returns
                int: Number of MCTS rollouts
            """
            return self._solver.get_nb_rollouts()

        def get_residual_moving_average(self) -> float:
            """Get the average Bellman error (residual) at the root state of the search,
                or an infinite value if the number of computed residuals is lower than
                the epsilon moving average window set in the MCTS instance's constructor

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
            Tuple[D.T_agent[D.T_concurrency[D.T_event]], float],
        ]:
            """Get the (partial) solution policy defined for the states for which
                the best value according to the execution action selector has been updated
                at least once (which is optimal if the algorithm has converged and labels are used)

            !!! warning
                Only defined over the states reachable from the last root solving state
                when node garbage was set to True in the MCTS instance's constructor

            # Returns
            Dict[ D.T_agent[D.T_observation], Tuple[D.T_agent[D.T_concurrency[D.T_event]], float], ]:
                Mapping from states to pairs of action and best value according to the
                execution action selector
            """
            return self._solver.get_policy()

        def get_action_prefix(self) -> List[D.T_agent[D.T_observation]]:
            """Get the list of actions returned by the solver so far after each
                call to the `MCTS.get_next_action` method (mostly internal use in order
                to rebuild the sequence of visited states until reaching the current
                solving state, when using `MCTS.TransitionMode.STEP` for which we can
                only progress the transition function with steps that hide the current
                state of the domain)

            # Returns
            List[D.T_agent[D.T_observation]]: List of actions executed by the solver
                so far after each call to the `MCTS.get_next_action` method
            """
            return self._solver.get_action_prefix()

    class HMCTS(MCTS):
        """MCTS solver to use with the multi-agent hierarchical `MAHD` solver
        as the multi-agent compound solver"""

        hyperparameters = [
            hp for hp in MCTS.hyperparameters if hp.name != "rollout_policy"
        ] + [
            IntegerHyperparameter(name="heuristic_confidence"),
            FloatHyperparameter(name="action_choice_noise"),
        ]

        def __init__(
            self,
            domain_factory: Callable[[], MCTS.T_domain],
            time_budget: int = 3600000,
            rollout_budget: int = 100000,
            max_depth: int = 1000,
            residual_moving_average_window: int = 100,
            epsilon: float = 0.0,  # not a stopping criterion by default
            discount: float = 1.0,
            ucb_constant: float = 1.0 / sqrt(2.0),
            online_node_garbage: bool = False,
            heuristic: Callable[
                [MCTS.T_domain, D.T_state],
                Tuple[
                    D.T_agent[Value[D.T_value]], D.T_agent[D.T_concurrency[D.T_event]]
                ],
            ] = None,
            heuristic_confidence: int = 1000,
            action_choice_noise: float = 0.1,
            state_expansion_rate: float = 0.1,
            action_expansion_rate: float = 0.1,
            transition_mode: MCTS.TransitionMode = MCTS.TransitionMode.DISTRIBUTION,
            tree_policy: MCTS.TreePolicy = MCTS.TreePolicy.DEFAULT,
            expander: MCTS.Expander = MCTS.Expander.FULL,
            action_selector_optimization: MCTS.ActionSelector = MCTS.ActionSelector.UCB1,
            action_selector_execution: MCTS.ActionSelector = MCTS.ActionSelector.BEST_Q_VALUE,
            back_propagator: MCTS.BackPropagator = MCTS.BackPropagator.GRAPH,
            continuous_planning: bool = True,
            parallel: bool = False,
            shared_memory_proxy=None,
            callback: Callable[
                [HMCTS, Optional[int]], bool
            ] = lambda slv, i=None: False,
            verbose: bool = False,
        ):
            """Construct a HMCTS solver instance

            # Parameters
            domain_factory (Callable[[], MCTS.T_domain]): The lambda function to create a domain instance.
            time_budget (int, optional): Maximum solving time in milliseconds. Defaults to 3600000.
            rollout_budget (int, optional): Maximum number of rollouts. Defaults to 100000.
            max_depth (int, optional): Maximum depth of each MCTS rollout. Defaults to 1000.
            residual_moving_average_window (int, optional): Number of latest computed residual values
                to memorize in order to compute the average Bellman error (residual) at the root state
                of the search. Defaults to 100.
            epsilon (float, optional): Maximum Bellman error (residual) allowed to decide that a state
                is solved, or to decide when no labels are used that the value function of the root state
                of the search has converged (in the latter case: the root state's Bellman error is averaged
                over the residual_moving_average_window). Defaults to 0.0.
            discount (float, optional): Value function's discount factor. Defaults to 1.0.
            ucb_constant (float, optional): UCB constant as used in the UCT algorithm when the action selector
                (for optimization or execution) is `MCTS.ActionSelector.UCB1`. Defaults to 1.0/sqrt(2.0).
            online_node_garbage (bool, optional): Boolean indicating whether the search graph which is
                no more reachable from the root solving state should be deleted (True) or not (False). Defaults to False.
            heuristic (Callable[ [MCTS.T_domain, D.T_state], Tuple[ D.T_agent[Value[D.T_value]], D.T_agent[D.T_concurrency[D.T_event]] ], ], optional):
                Multi-agent compound heuristic as returned by the `MAHD` algorithm from independent
                agent heuristic contributions. Defaults to None (no heuristic in use).
            heuristic_confidence (int, optional): Fake state node visits set on non-expanded state nodes for which the
                multi-agent compound heuristic is computed by `MAHD`. Defaults to 1000.
            action_choice_noise (float, optional): Probability used to sample random actions instead of executing the
                compound heuristic actions returned by the `MAHD` algorithm. Defaults to 0.1.
            state_expansion_rate (float, optional): Value $rs$ used when the expander is `MCTS.Expander.PARTIAL`
                such that the probability of discovering new applicable actions in a given state node with already $na$ discovered
                applicable actions is equal to $e^{-rs \cdot na}$. Defaults to 0.1.
            action_expansion_rate (float, optional):  Value $ra$ used when the expander is `MCTS.Expander.PARTIAL`
                such that the probability of discovering new state outcomes in a given action node with already $ns$ discovered
                state outcomes is equal to $e^{-ra \cdot ns}$. Defaults to 0.1.
            transition_mode (MCTS.TransitionMode, optional): Transition mode enum (one of `MCTS.TransitionMode.STEP`,
                `MCTS.TransitionMode.SAMPLE` or `MCTS.TransitionMode.DISTRIBUTION` to progress the
                trajectories with, respectively, the 'step' or 'sample' or 'get_next_state_distribution' method of the domain
                depending on the domain's dynamics capabilities). Defaults to `MCTS.TransitionMode.DISTRIBUTION`.
            tree_policy (MCTS.TreePolicy, optional): Tree policy enum (currently only
                `MCTS.TreePolicy.DEFAULT` which rollouts a random trajectory from the current root
                solving state until reaching a non-expanded state node of the tree). Defaults to `MCTS.TreePolicy.DEFAULT`.
            expander (MCTS.Expander, optional): Expander enum used when a state needs to be
                expanded (one of `MCTS.Expander.FULL` if applicable actions and next states should be
                all enumerated for each transition function, or `MCTS.Expander.PARTIAL` if they should
                be sampled with a probability which exponentially decreases as the number of already discovered
                applicable actions and next states increases). Defaults to `MCTS.Expander.FULL`.
            action_selector_optimization (MCTS.ActionSelector, optional): Action selector class used to
                select actions in the tree policy's trajectory simulations (one of
                `MCTS.ActionSelector.UCB1` to select the action based on the UCB criterion, or
                `MCTS.ActionSelector.BEST_Q_VALUE` to select the action with maximum Q-Value in the
                current state node). Defaults to `MCTS.ActionSelector.UCB1`.
            action_selector_execution (MCTS.ActionSelector, optional): Action selector class used to
                select actions at execution time when the 'get_best_action' method of the
                solver is invoked in a given execution state (one of
                `MCTS.ActionSelector.UCB1` to select the action based on the UCB criterion, or
                `MCTS.ActionSelector.BEST_Q_VALUE` to select the action with maximum Q-Value in the
                current state node). Defaults to `MCTS.ActionSelector.BEST_Q_VALUE`.
            back_propagator (MCTS.BackPropagator, optional): Back propagator enum (currently only
                `MCTS.BackPropagator.GRAPH` which back-propagates empirical Q-values from non-expanded
                state nodes up to the root node of the tree along the tree policy's sampled
                trajectories). Defaults to `MCTS.BackPropagator.GRAPH`.
            continuous_planning (bool, optional): Boolean whether the solver should optimize again the policy
                from the current solving state (True) or not (False) even if the policy is already defined
                in this state. Defaults to True.
            parallel (bool, optional): Parallelize MCTS rollouts on different processes using duplicated domains (True)
                or not (False). Defaults to False.
            shared_memory_proxy (_type_, optional): The optional shared memory proxy. Defaults to None.
            callback (Callable[[HMCTS, Optional[int]], optional): Function called at the end of each MCTS rollout,
                taking as arguments the solver and the thread/process ID (i.e. parallel domain ID, which is equal to None
                in case of sequential execution, i.e. when 'parallel' is set to False in this constructor) from
                which the callback is called, and returning True if the solver must be stopped. The callback lambda
                function cannot take the (potentially parallelized) domain as argument because we could not otherwise
                serialize (i.e. pickle) the solver to pass it to the corresponding parallel domain process in case of parallel
                execution. Nevertheless, the `ParallelSolver.get_domain` method callable on the solver instance
                can be used to retrieve either the user domain in sequential execution, or the parallel domains proxy
                `ParallelDomain` in parallel execution from which domain methods can be called by using the
                callback's process ID argument. Defaults to (lambda slv, i=None: False).
            verbose (bool, optional): Boolean indicating whether verbose messages should be logged (True)
                or not (False). Defaults to False.
            """
            super().__init__(
                domain_factory=domain_factory,
                time_budget=time_budget,
                rollout_budget=rollout_budget,
                max_depth=max_depth,
                residual_moving_average_window=residual_moving_average_window,
                epsilon=epsilon,
                discount=discount,
                ucb_constant=ucb_constant,
                online_node_garbage=online_node_garbage,
                heuristic=lambda d, o: self._value_heuristic(d, o),
                custom_policy=lambda d, o: self._policy_heuristic(d, o),
                state_expansion_rate=state_expansion_rate,
                action_expansion_rate=action_expansion_rate,
                transition_mode=transition_mode,
                tree_policy=tree_policy,
                expander=expander,
                action_selector_optimization=action_selector_optimization,
                action_selector_execution=action_selector_execution,
                rollout_policy=MCTS.RolloutPolicy.CUSTOM,
                back_propagator=back_propagator,
                continuous_planning=continuous_planning,
                parallel=parallel,
                shared_memory_proxy=shared_memory_proxy,
                callback=callback,
                verbose=verbose,
            )
            self._compound_heuristic = heuristic
            self._heuristic_confidence = heuristic_confidence
            self._action_choice_noise = action_choice_noise
            self._heuristic_records = {}

        def _reset(self) -> None:
            """Clears the search graph and the heuristic records."""
            super()._reset()
            self._heuristic_records = {}

        def _value_heuristic(
            self, domain: MCTS.T_domain, observation: D.T_agent[D.T_observation]
        ) -> Tuple[D.T_agent[Value[D.T_value]], int]:
            """Reconstitutes the MCTS heuristic used to initialize the value of non-expanded
                state nodes from the multi-agent compound heuristic computed by the
                `MAHD` algorithm

            # Parameters
            domain (MCTS.T_domain): The domain instance
            observation (D.T_agent[D.T_observation]): The non-expanded state node from which
                the heuristic must be computed

            # Returns
            Tuple[D.T_agent[Value[D.T_value]], int]: MCTS heuristic value at the given state
            """
            if observation not in self._heuristic_records:
                self._heuristic_records[observation] = self._compound_heuristic(
                    domain, observation
                )
            return (self._heuristic_records[observation][0], self._heuristic_confidence)

        def _policy_heuristic(
            self, domain: MCTS.T_domain, observation: D.T_agent[D.T_observation]
        ) -> D.T_agent[D.T_concurrency[D.T_event]]:
            """Reconstitutes the MCTS custom rollout executed starting in non-expanded
                state nodes from the multi-agent compound heuristic computed by the
                `MAHD` algorithm

            # Parameters
            domain (MCTS.T_domain): The domain instance
            observation (D.T_agent[D.T_observation]): The non-expanded state node from which
                the custom rollout policy must be run

            # Returns
            D.T_agent[D.T_concurrency[D.T_event]]: Rollout action to execute in the given state
            """
            if observation not in self._heuristic_records:
                self._heuristic_records[observation] = self._compound_heuristic(
                    domain, observation
                )
            if rd.random() > self._action_choice_noise:
                return self._heuristic_records[observation][1]
            else:
                return domain.get_applicable_actions(observation).sample()

    class UCT(MCTS):
        """UCT as described in the paper " Bandit Based Monte-Carlo Planning" by
        Levente Kocsis and Csaba Szepesvari (ECML 2006) is a famous variant of MCTS
        with some specific options including the famous UCB1 action selector to perform tree exploration
        """

        hyperparameters = [
            hp
            for hp in MCTS.hyperparameters
            if hp.name
            not in {
                "state_expansion_rate",
                "action_expansion_rate",
                "tree_policy",
                "expander",
                "action_selector_optimization",
                "action_selector_execution",
                "back_propagator",
            }
        ]

        def __init__(
            self,
            domain_factory: Callable[[], MCTS.T_domain],
            time_budget: int = 3600000,
            rollout_budget: int = 100000,
            max_depth: int = 1000,
            residual_moving_average_window: int = 100,
            epsilon: float = 0.0,  # not a stopping criterion by default
            discount: float = 1.0,
            ucb_constant: float = 1.0 / sqrt(2.0),
            online_node_garbage: bool = False,
            custom_policy: Callable[
                [MCTS.T_domain, D.T_agent[D.T_observation]],
                D.T_agent[D.T_concurrency[D.T_event]],
            ] = None,
            heuristic: Callable[
                [MCTS.T_domain, D.T_agent[D.T_observation]],
                Tuple[D.T_agent[Value[D.T_value]], int],
            ] = None,
            transition_mode: MCTS.TransitionMode = MCTS.TransitionMode.DISTRIBUTION,
            rollout_policy: MCTS.RolloutPolicy = MCTS.RolloutPolicy.RANDOM,
            continuous_planning: bool = True,
            parallel: bool = False,
            shared_memory_proxy=None,
            callback: Callable[[UCT, Optional[int]], bool] = lambda slv, i=None: False,
            verbose: bool = False,
        ) -> None:
            """Construct a UCT solver instance

            # Parameters
            domain_factory (Callable[[], MCTS.T_domain]): The lambda function to create a domain instance.
            time_budget (int, optional): Maximum solving time in milliseconds. Defaults to 3600000.
            rollout_budget (int, optional): Maximum number of rollouts. Defaults to 100000.
            max_depth (int, optional): Maximum depth of each UCT rollout. Defaults to 1000.
            residual_moving_average_window (int, optional): Number of latest computed residual values
                to memorize in order to compute the average Bellman error (residual) at the root state
                of the search. Defaults to 100.
            epsilon (float, optional): Maximum Bellman error (residual) allowed to decide that a state
                is solved, or to decide when no labels are used that the value function of the root state
                of the search has converged (in the latter case: the root state's Bellman error is averaged
                over the residual_moving_average_window). Defaults to 0.0.
            discount (float, optional): Value function's discount factor. Defaults to 1.0.
            ucb_constant (float, optional): UCB constant as used in the UCT algorithm when the action selector
                (for optimization or execution) is `MCTS.ActionSelector.UCB1`. Defaults to 1.0/sqrt(2.0).
            online_node_garbage (bool, optional): Boolean indicating whether the search graph which is
                no more reachable from the root solving state should be deleted (True) or not (False). Defaults to False.
            custom_policy (Callable[ [MCTS.T_domain, D.T_agent[D.T_observation]], D.T_agent[D.T_concurrency[D.T_event]], ], optional):
                Custom policy function to use in the rollout policy from non-expanded state nodes when the rollout policy is
                `MCTS.RolloutPolicy.CUSTOM`. Defaults to None (no custom policy in use).
            heuristic (Callable[ [MCTS.T_domain, D.T_agent[D.T_observation]], Tuple[D.T_agent[Value[D.T_value]], int], ], optional):
                Optional Heuristic function to initialize non-expanded state nodes (returns a pair of value estimate and
                fake number of visit counts). Defaults to None (no heuristic in use).
            transition_mode (MCTS.TransitionMode, optional): Transition mode enum (one of `MCTS.TransitionMode.STEP`,
                `MCTS.TransitionMode.SAMPLE` or `MCTS.TransitionMode.DISTRIBUTION` to progress the
                trajectories with, respectively, the 'step' or 'sample' or 'get_next_state_distribution' method of the domain
                depending on the domain's dynamics capabilities). Defaults to `MCTS.TransitionMode.DISTRIBUTION`.
            rollout_policy (MCTS.RolloutPolicy, optional): Rollout policy enum (one of
                :`MCTS.RolloutPolicy.RANDOM` to simulate trajectories starting in a non-expanded state
                node of the tree by sampling random applicable actions in each visited state, or
                `MCTS.RolloutPolicy.CUSTOM` to simulate them by applying actions from the given policy
                'custom_policy' given to this constructor, or `MCTS.RolloutPolicy.VOID` to deactivate
                the simulation of trajectories from non-expanded state nodes, in which latter case it is advised to
                provide the 'heuristic' function in this constructor to initialize non-expanded state nodes' values).
                Defaults to :`MCTS.RolloutPolicy.RANDOM`.
            continuous_planning (bool, optional): Boolean whether the solver should optimize again the policy
                from the current solving state (True) or not (False) even if the policy is already defined
                in this state. Defaults to True.
            parallel (bool, optional): Parallelize MCTS rollouts on different processes using duplicated domains (True)
                or not (False). Defaults to False.
            shared_memory_proxy (_type_, optional): The optional shared memory proxy. Defaults to None.
            callback (Callable[[UCT, Optional[int]], optional): Function called at the end of each RIW rollout,
                taking as arguments the solver and the thread/process ID (i.e. parallel domain ID, which is equal to None
                in case of sequential execution, i.e. when 'parallel' is set to False in this constructor) from
                which the callback is called, and returning True if the solver must be stopped. The callback lambda
                function cannot take the (potentially parallelized) domain as argument because we could not otherwise
                serialize (i.e. pickle) the solver to pass it to the corresponding parallel domain process in case of parallel
                execution. Nevertheless, the `ParallelSolver.get_domain` method callable on the solver instance
                can be used to retrieve either the user domain in sequential execution, or the parallel domains proxy
                `ParallelDomain` in parallel execution from which domain methods can be called by using the
                callback's process ID argument. Defaults to (lambda slv, i=None: False).
            verbose (bool, optional): Boolean indicating whether verbose messages should be logged (True)
                or not (False). Defaults to False.
            """
            super().__init__(
                domain_factory=domain_factory,
                time_budget=time_budget,
                rollout_budget=rollout_budget,
                max_depth=max_depth,
                residual_moving_average_window=residual_moving_average_window,
                epsilon=epsilon,
                discount=discount,
                ucb_constant=ucb_constant,
                online_node_garbage=online_node_garbage,
                custom_policy=custom_policy,
                heuristic=heuristic,
                transition_mode=transition_mode,
                tree_policy=MCTS.TreePolicy.DEFAULT,
                expander=MCTS.Expander.FULL,
                action_selector_optimization=MCTS.ActionSelector.UCB1,
                action_selector_execution=MCTS.ActionSelector.BEST_Q_VALUE,
                rollout_policy=rollout_policy,
                back_propagator=MCTS.BackPropagator.GRAPH,
                continuous_planning=continuous_planning,
                parallel=parallel,
                shared_memory_proxy=shared_memory_proxy,
                callback=callback,
                verbose=verbose,
            )

    class HUCT(HMCTS):
        """UCT solver to use with the multi-agent hierarchical `MAHD` solver
        as the multi-agent compound solver"""

        hyperparameters = [
            hp
            for hp in HMCTS.hyperparameters
            if hp.name
            not in {
                "state_expansion_rate",
                "action_expansion_rate",
                "tree_policy",
                "expander",
                "action_selector_optimization",
                "action_selector_execution",
                "back_propagator",
            }
        ]

        def __init__(
            self,
            domain_factory: Callable[[], MCTS.T_domain],
            time_budget: int = 3600000,
            rollout_budget: int = 100000,
            max_depth: int = 1000,
            residual_moving_average_window: int = 100,
            epsilon: float = 0.0,  # not a stopping criterion by default
            discount: float = 1.0,
            ucb_constant: float = 1.0 / sqrt(2.0),
            online_node_garbage: float = False,
            heuristic: Callable[
                [MCTS.T_domain, D.T_state],
                Tuple[
                    D.T_agent[Value[D.T_value]], D.T_agent[D.T_concurrency[D.T_event]]
                ],
            ] = None,
            heuristic_confidence: int = 1000,
            action_choice_noise: float = 0.1,
            transition_mode: MCTS.TransitionMode = MCTS.TransitionMode.DISTRIBUTION,
            continuous_planning: bool = True,
            parallel: bool = False,
            shared_memory_proxy=None,
            callback: Callable[[HUCT, Optional[int]], bool] = lambda slv, i=None: False,
            verbose: bool = False,
        ) -> None:
            """Construct a HUCT solver instance

            # Parameters
            domain_factory (Callable[[], MCTS.T_domain]): The lambda function to create a domain instance.
            time_budget (int, optional): Maximum solving time in milliseconds. Defaults to 3600000.
            rollout_budget (int, optional): Maximum number of rollouts. Defaults to 100000.
            max_depth (int, optional): Maximum depth of each UCT rollout. Defaults to 1000.
            residual_moving_average_window (int, optional): Number of latest computed residual values
                to memorize in order to compute the average Bellman error (residual) at the root state
                of the search. Defaults to 100.
            epsilon (float, optional): Maximum Bellman error (residual) allowed to decide that a state
                is solved, or to decide when no labels are used that the value function of the root state
                of the search has converged (in the latter case: the root state's Bellman error is averaged
                over the residual_moving_average_window). Defaults to 0.0.
            discount (float, optional): Value function's discount factor. Defaults to 1.0.
            ucb_constant (float, optional): UCB constant as used in the UCT algorithm when the action selector
                (for optimization or execution) is `MCTS.ActionSelector.UCB1`. Defaults to 1.0/sqrt(2.0).
            online_node_garbage (bool, optional): Boolean indicating whether the search graph which is
                no more reachable from the root solving state should be deleted (True) or not (False). Defaults to False.
            heuristic (Callable[ [MCTS.T_domain, D.T_state], Tuple[ D.T_agent[Value[D.T_value]], D.T_agent[D.T_concurrency[D.T_event]] ], ], optional):
                Multi-agent compound heuristic as returned by the `MAHD` algorithm from independent
                agent heuristic contributions. Defaults to None (no heuristic in use).
            heuristic_confidence (int, optional): Fake state node visits set on non-expanded state nodes for which the
                multi-agent compound heuristic is computed by `MAHD`. Defaults to 1000.
            action_choice_noise (float, optional): Probability used to sample random actions instead of executing the
                compound heuristic actions returned by the `MAHD` algorithm. Defaults to 0.1.
            transition_mode (MCTS.TransitionMode, optional): Transition mode enum (one of `MCTS.TransitionMode.STEP`,
                `MCTS.TransitionMode.SAMPLE` or `MCTS.TransitionMode.DISTRIBUTION` to progress the
                trajectories with, respectively, the 'step' or 'sample' or 'get_next_state_distribution' method of the domain
                depending on the domain's dynamics capabilities). Defaults to `MCTS.TransitionMode.DISTRIBUTION`.
            continuous_planning (bool, optional): Boolean whether the solver should optimize again the policy
                from the current solving state (True) or not (False) even if the policy is already defined
                in this state. Defaults to True.
            parallel (bool, optional): Parallelize MCTS rollouts on different processes using duplicated domains (True)
                or not (False). Defaults to False.
            shared_memory_proxy (_type_, optional): The optional shared memory proxy. Defaults to None.
            callback (Callable[[HUCT, Optional[int]], optional): Function called at the end of each RIW rollout,
                taking as arguments the solver and the thread/process ID (i.e. parallel domain ID, which is equal to None
                in case of sequential execution, i.e. when 'parallel' is set to False in this constructor) from
                which the callback is called, and returning True if the solver must be stopped. The callback lambda
                function cannot take the (potentially parallelized) domain as argument because we could not otherwise
                serialize (i.e. pickle) the solver to pass it to the corresponding parallel domain process in case of parallel
                execution. Nevertheless, the `ParallelSolver.get_domain` method callable on the solver instance
                can be used to retrieve either the user domain in sequential execution, or the parallel domains proxy
                `ParallelDomain` in parallel execution from which domain methods can be called by using the
                callback's process ID argument. Defaults to (lambda slv, i=None: False).
            verbose (bool, optional): Boolean indicating whether verbose messages should be logged (True)
                or not (False). Defaults to False.
            """
            super().__init__(
                domain_factory=domain_factory,
                time_budget=time_budget,
                rollout_budget=rollout_budget,
                max_depth=max_depth,
                residual_moving_average_window=residual_moving_average_window,
                epsilon=epsilon,
                discount=discount,
                ucb_constant=ucb_constant,
                online_node_garbage=online_node_garbage,
                heuristic=heuristic,
                heuristic_confidence=heuristic_confidence,
                action_choice_noise=action_choice_noise,
                transition_mode=transition_mode,
                tree_policy=MCTS.TreePolicy.DEFAULT,
                expander=MCTS.Expander.FULL,
                action_selector_optimization=MCTS.ActionSelector.UCB1,
                action_selector_execution=MCTS.ActionSelector.BEST_Q_VALUE,
                back_propagator=MCTS.BackPropagator.GRAPH,
                continuous_planning=continuous_planning,
                parallel=parallel,
                shared_memory_proxy=shared_memory_proxy,
                callback=callback,
                verbose=verbose,
            )

except ImportError:
    sys.path = record_sys_path
    print(
        'Scikit-decide C++ hub library not found. Please check it is installed in "skdecide/hub".'
    )
    raise
