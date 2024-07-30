# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import random
from enum import Enum
from typing import Callable

import networkx as nx
import numpy as np
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    EnumHyperparameter,
)

from skdecide import Domain
from skdecide.builders.domain.scheduling.scheduling_domains import SchedulingDomain
from skdecide.builders.domain.scheduling.scheduling_domains_modelling import (
    SchedulingAction,
    SchedulingActionEnum,
    State,
)
from skdecide.solvers import DeterministicPolicies, Solver

D = SchedulingDomain


class GreedyChoice(Enum):
    """Enumeration representing different greedy method to use in PilePolicy"""

    MOST_SUCCESSORS = 1
    SAMPLE_MOST_SUCCESSORS = 2
    FASTEST = 3
    TOTALLY_RANDOM = 4


GreedyChoice.MOST_SUCCESSORS.__doc__ = (
    "Start first the tasks that have the most successors in the precedence graph"
)
GreedyChoice.SAMPLE_MOST_SUCCESSORS.__doc__ = "Sample next task to schedule based on a probability weight proportional to its number of successors in the precedence graph"
GreedyChoice.FASTEST.__doc__ = "Schedule first the task that has the lowest duration"
GreedyChoice.TOTALLY_RANDOM.__doc__ = "Sample random next task to schedule next"


class PilePolicy(Solver, DeterministicPolicies):
    T_domain = D

    hyperparameters = [
        EnumHyperparameter(
            name="greedy_method",
            enum=GreedyChoice,
        ),
    ]

    def __init__(
        self,
        domain_factory: Callable[[], Domain],
        greedy_method: GreedyChoice = GreedyChoice.MOST_SUCCESSORS,
    ):
        """Greedy Pile/Queue based solver for scheduling problems.

        This solver/policy is greedily scheduling task based on some rule specified by GreedyChoice enumerator.
        The resulting solution is not insured to respect specific constraints/needs for the scheduling problem.

        # Parameters
        domain_factory: A callable with no argument returning the domain to solve (can be a mere domain class).
            The resulting domain will be auto-cast to the level expected by the solver.
        greedy_method : Greedy method to use.

        """
        Solver.__init__(self, domain_factory=domain_factory)
        self.greedy_method = greedy_method

    def _solve(self) -> None:
        self.domain = self._domain_factory()
        self.graph = self.domain.graph
        self.nx_graph: nx.DiGraph = self.graph.to_networkx()
        self.successors_map = {}
        self.predecessors_map = {}
        # successors = nx.dfs_successors(self.nx_graph, 1, self.n_jobs+2)
        self.successors = {
            n: list(nx.algorithms.descendants(self.nx_graph, n))
            for n in self.nx_graph.nodes()
        }
        self.source = 1
        for k in self.successors:
            self.successors_map[k] = {
                "succs": self.successors[k],
                "nb": len(self.successors[k]),
            }
        self.predecessors = {
            n: list(nx.algorithms.ancestors(self.nx_graph, n))
            for n in self.nx_graph.nodes()
        }
        for k in self.predecessors:
            self.predecessors_map[k] = {
                "succs": self.predecessors[k],
                "nb": len(self.predecessors[k]),
            }

    def _get_next_action(
        self, observation: D.T_agent[D.T_observation]
    ) -> D.T_agent[D.T_concurrency[D.T_event]]:
        s: State = observation
        predecessors = {
            n: nx.algorithms.ancestors(self.nx_graph, n) for n in self.nx_graph.nodes()
        }
        for k in predecessors:
            self.predecessors_map[k] = {
                "succs": predecessors[k],
                "nb": len(predecessors[k]),
            }
        greedy_choice = self.greedy_method
        possible_task_to_launch = self.domain.task_possible_to_launch_precedence(
            state=s
        )
        possible_task_to_launch = [
            t
            for t in possible_task_to_launch
            if self.domain.check_if_action_can_be_started(
                state=s,
                action=SchedulingAction(
                    task=t,
                    action=SchedulingActionEnum.START,
                    time_progress=False,
                    mode=1,
                ),
            )[0]
        ]
        if len(possible_task_to_launch) > 0:
            if greedy_choice == GreedyChoice.MOST_SUCCESSORS:
                next_activity = max(
                    possible_task_to_launch, key=lambda x: self.successors_map[x]["nb"]
                )
            if greedy_choice == GreedyChoice.SAMPLE_MOST_SUCCESSORS:
                prob = np.array(
                    [
                        self.successors_map[possible_task_to_launch[i]]["nb"]
                        for i in range(len(possible_task_to_launch))
                    ]
                )
                s = np.sum(prob)
                if s != 0:
                    prob = prob / s
                else:
                    prob = (
                        1.0
                        / len(possible_task_to_launch)
                        * np.ones((len(possible_task_to_launch)))
                    )
                next_activity = np.random.choice(
                    np.arange(0, len(possible_task_to_launch)), size=1, p=prob
                )[0]
                next_activity = possible_task_to_launch[next_activity]
            if greedy_choice == GreedyChoice.FASTEST:
                next_activity = min(
                    possible_task_to_launch,
                    key=lambda x: self.domain.sample_task_duration(x, 1, 0.0),
                )
            if greedy_choice == GreedyChoice.TOTALLY_RANDOM:
                next_activity = random.choice(possible_task_to_launch)
            return SchedulingAction(
                task=next_activity,
                mode=1,
                action=SchedulingActionEnum.START,
                time_progress=False,
            )
        else:
            return SchedulingAction(
                task=None,
                mode=1,
                action=SchedulingActionEnum.TIME_PR,
                time_progress=True,
            )

    def _is_policy_defined_for(self, observation: D.T_agent[D.T_observation]) -> bool:
        return True
