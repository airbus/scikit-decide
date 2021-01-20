from typing import Callable
from skdecide.solvers import Solver, DeterministicPolicies
from skdecide.builders.scheduling.scheduling_domains_modelling import State, SchedulingAction, SchedulingActionEnum
from skdecide.builders.scheduling.scheduling_domains import SchedulingDomain
from skdecide.builders.scheduling.modes import SingleMode, MultiMode
from enum import Enum
import numpy as np
import random
import networkx as nx
D = SchedulingDomain


class GreedyChoice(Enum):
    MOST_SUCCESSORS = 1
    SAMPLE_MOST_SUCCESSORS = 2
    FASTEST = 3
    TOTALLY_RANDOM = 4


class PilePolicy(Solver, DeterministicPolicies):
    T_domain = D

    def __init__(self, greedy_method: GreedyChoice=GreedyChoice.MOST_SUCCESSORS):
        self.greedy_method = greedy_method

    def _solve_domain(self, domain_factory: Callable[[], D]) -> None:
        self.domain = domain_factory()
        self.graph = self.domain.graph
        self.nx_graph: nx.DiGraph = self.graph.to_networkx()
        self.successors_map = {}
        self.predecessors_map = {}
        # successors = nx.dfs_successors(self.nx_graph, 1, self.n_jobs+2)
        self.successors = {n: list(nx.algorithms.descendants(self.nx_graph, n))
                           for n in self.nx_graph.nodes()}
        self.source = 1
        for k in self.successors:
            self.successors_map[k] = {"succs": self.successors[k], "nb": len(self.successors[k])}
        self.predecessors = {n: list(nx.algorithms.ancestors(self.nx_graph, n))
                             for n in self.nx_graph.nodes()}
        for k in self.predecessors:
            self.predecessors_map[k] = {"succs": self.predecessors[k], "nb": len(self.predecessors[k])}

    def _get_next_action(self, observation: D.T_agent[D.T_observation]) -> D.T_agent[D.T_concurrency[D.T_event]]:
        s: State = observation
        predecessors = {n: nx.algorithms.ancestors(self.nx_graph, n)
                        for n in self.nx_graph.nodes()}
        for k in predecessors:
            self.predecessors_map[k] = {"succs": predecessors[k], "nb": len(predecessors[k])}
        greedy_choice = self.greedy_method
        possible_task_to_launch = self.domain.task_possible_to_launch_precedence(state=s)
        possible_task_to_launch = [t
                                   for t in possible_task_to_launch if
                                   self.domain.check_if_action_can_be_started(state=s,
                                                                              action=SchedulingAction(task=t,
                                                                                                      action=
                                                                                             SchedulingActionEnum.START,
                                                                                                      time_progress=False,
                                                                                                      mode=1))[0]
                                   ]
        if len(possible_task_to_launch) > 0:
            if greedy_choice == GreedyChoice.MOST_SUCCESSORS:
                next_activity = max(possible_task_to_launch,
                                    key=lambda x: self.successors_map[x]["nb"])
            if greedy_choice == GreedyChoice.SAMPLE_MOST_SUCCESSORS:
                prob = np.array([self.successors_map[possible_task_to_launch[i]]["nb"]
                                 for i in range(len(possible_task_to_launch))])
                s = np.sum(prob)
                if s != 0:
                    prob = prob / s
                else:
                    prob = 1. / len(possible_task_to_launch) * np.ones((len(possible_task_to_launch)))
                next_activity = np.random.choice(np.arange(0,
                                                           len(possible_task_to_launch)), size=1,
                                                 p=prob)[0]
                next_activity = possible_task_to_launch[next_activity]
            if greedy_choice == GreedyChoice.FASTEST:
                next_activity = min(possible_task_to_launch,
                                    key=lambda x: self.domain.sample_task_duration(x, 1, 0.))
            if greedy_choice == GreedyChoice.TOTALLY_RANDOM:
                next_activity = random.choice(possible_task_to_launch)
            return SchedulingAction(task=next_activity,
                                    mode=1,
                                    action=SchedulingActionEnum.START,
                                    time_progress=False)
        else:
            return SchedulingAction(task=None, mode=1,
                                    action=SchedulingActionEnum.TIME_PR,
                                    time_progress=True)

    def _is_policy_defined_for(self, observation: D.T_agent[D.T_observation]) -> bool:
        return True



