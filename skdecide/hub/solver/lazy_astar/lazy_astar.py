# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from heapq import heappush, heappop
from itertools import count
from typing import Optional, Callable

from skdecide import Domain, Solver
from skdecide.builders.domain import SingleAgent, Sequential, DeterministicTransitions, Actions, Goals, Markovian, \
    FullyObservable, PositiveCosts
from skdecide.builders.solver import DeterministicPolicies, Utilities


# TODO: remove Markovian req?
class D(Domain, SingleAgent, Sequential, DeterministicTransitions, Actions, Goals, Markovian, FullyObservable,
        PositiveCosts):
    pass


class LazyAstar(Solver, DeterministicPolicies, Utilities):
    T_domain = D

    def __init__(self, from_state: Optional[D.T_state] = None,
                 heuristic: Optional[Callable[[D.T_state, Domain], float]] = None, weight: float = 1.,
                 verbose: bool = False, render: bool = False) -> None:

        self._from_state = from_state
        self._heuristic = (lambda _, __: 0.) if heuristic is None else heuristic
        self._weight = weight
        self._verbose = verbose
        self._render = render
        self._values = {}
        self._plan = []

    def _solve_domain(self, domain_factory: Callable[[], D]) -> None:
        self._domain = domain_factory()

        def extender(node, label, explored):
            neigh = [(self._domain.get_next_state(node, a), a)
                     for a in self._domain.get_applicable_actions(node).get_elements()]
            neigh_not_explored = [(n, a) for n, a in neigh if n not in explored]
            cost_labels = [(n, self._domain.get_transition_value(node, a, n).cost, {'action': a})
                           for n, a in neigh_not_explored]
            return cost_labels

        push = heappush
        pop = heappop
        if self._from_state is None:
            # get initial observation from domain (assuming DeterministicInitialized)
            sources = [self._domain.get_initial_state()]
        else:
            sources = [self._from_state]
        # targets = list(self._domain.get_goals().get_elements())

        # The queue is the OPEN list.
        # It stores priority, node, cost to reach, parent and label (any data type) of transition from parent.
        # Uses Python heapq to keep in priority order.
        # Add a counter to the queue to prevent the underlying heap from
        # attempting to compare the nodes themselves. The hash breaks ties in the
        # priority and is guaranteed unique for all nodes in the graph.
        c = count()

        # TODO: check if necessary (a priori used to keep additional infos)
        initial_label = {source: None for source in sources}
        # Maps enqueued nodes to distance of discovered paths and the
        # computed heuristics to target. We avoid computing the heuristics
        # more than once and inserting the node into the queue too many times.
        enqueued = {source: (0, self._weight * self._heuristic(source, self._domain)) for source in sources}
        # enqueued = {source: min([(0, self._weight * self._heuristic(source, target, initial_label[source]))
        # for target in targets], key=lambda x: x[1]) for source in sources}
        queue = [(enqueued[source][1], next(c), source, 0, None, initial_label[source]) for source in sources]
        # The explored dict is the CLOSED list.
        # It maps explored nodes to a pair of parent closest to the source and label of transition from parent.
        explored = {}
        path = []
        estim_total = 0.
        while queue:
            # Pop the smallest item from queue, i.e. with smallest f-value
            estim_total, __, curnode, dist, parent, label = pop(queue)
            if self._render:
                self._domain.render(curnode)
            if self._verbose:
                print(curnode, f'- cumulated cost: {dist} - estimated total cost: {estim_total}')
            if self._domain.is_goal(curnode):
                path = [(parent, label),
                        (curnode, None)]
                node = parent
                while node is not None:
                    (parent, label) = explored[node]
                    if parent is not None:
                        path.insert(0, (parent, label))
                    node = parent
                break  # return path, dist, enqueued[curnode][0], len(enqueued)
            if curnode in explored:
                continue
            explored[curnode] = (parent, label)
            for neighbor, cost, lbl in extender(curnode, label, explored):
                if neighbor in explored:
                    continue
                ncost = dist + cost
                if neighbor in enqueued:
                    qcost, h = enqueued[neighbor]
                    # if qcost < ncost, a longer path to neighbor remains
                    # enqueued. Removing it would need to filter the whole
                    # queue, it's better just to leave it there and ignore
                    # it when we visit the node a second time.
                    if qcost <= ncost:
                        continue
                else:
                    # h = min([self._heuristic(neighbor, target, lbl) for target in targets])
                    h = self._heuristic(neighbor, self._domain)
                enqueued[neighbor] = ncost, h
                push(queue, (ncost + (self._weight * h), next(c), neighbor, ncost, curnode, lbl))
        self._policy = {}
        for node, label in path:
            self._policy[node] = label['action'] if label is not None else None
            self._values[node] = estim_total - enqueued[node][0]
            if self._policy[node] is not None:
                self._plan.append(self._policy[node])
        # return estim_total, path  # TODO: find a way to expose these things through public API?

    def _get_next_action(self, observation: D.T_agent[D.T_observation]) -> D.T_agent[D.T_concurrency[D.T_event]]:
        return self._policy[observation]

    def _is_policy_defined_for(self, observation: D.T_agent[D.T_observation]) -> bool:
        return observation in self._policy

    def _get_utility(self, observation: D.T_agent[D.T_observation]) -> D.T_value:
        if observation not in self._values:
            return self._heuristic(observation, self._domain)
        return self._values[observation]
