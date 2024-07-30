# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from heapq import heappop, heappush
from itertools import count
from typing import Callable, Dict, List, Optional

from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    FloatHyperparameter,
)

from skdecide import Domain, Solver, Value
from skdecide.builders.domain import (
    Actions,
    DeterministicTransitions,
    FullyObservable,
    Goals,
    Markovian,
    PositiveCosts,
    Sequential,
    SingleAgent,
)
from skdecide.builders.solver import DeterministicPolicies, FromAnyState, Utilities


# TODO: remove Markovian req?
class D(
    Domain,
    SingleAgent,
    Sequential,
    DeterministicTransitions,
    Actions,
    Goals,
    Markovian,
    FullyObservable,
    PositiveCosts,
):
    pass


class LazyAstar(Solver, DeterministicPolicies, Utilities, FromAnyState):
    """Lazy A* solver."""

    T_domain = D

    hyperparameters = [
        FloatHyperparameter(
            name="weight",
            low=0.0,
            high=1.0,
            suggest_high=True,
            suggest_low=True,
        )
    ]

    def __init__(
        self,
        domain_factory: Callable[[], Domain],
        heuristic: Optional[
            Callable[[Domain, D.T_state], D.T_agent[Value[D.T_value]]]
        ] = None,
        weight: float = 1.0,
        verbose: bool = False,
        render: bool = False,
        callback: Callable[[LazyAstar], bool] = lambda solver: False,
    ) -> None:
        """

        # Parameters
        domain_factory
        heuristic
        weight
        verbose
        render
        callback: function called at each solver iteration. If returning true, the solve process stops.

        """
        self.callback = callback
        Solver.__init__(self, domain_factory=domain_factory)
        self._domain = self._domain_factory()
        self._heuristic = (
            (lambda _, __: Value(cost=0.0)) if heuristic is None else heuristic
        )
        self._weight = weight
        self._verbose = verbose
        self._render = render
        self._values = {}
        self._plan: List[D.T_event] = []

    def get_plan(self) -> List[D.T_event]:
        """Return the computed plan."""
        return self._plan

    def get_policy(self) -> Dict[D.T_observation, Optional[D.T_event]]:
        """Return the computed policy."""
        return self._policy

    def _solve_from(
        self,
        memory: D.T_state,
    ) -> None:
        """Run the solving process from a given state.

        # Parameters
        memory: The source memory (state or history) of the transition.

        !!! tip
            The nature of the solutions produced here depends on other solver's characteristics like
            #policy and #assessibility.
        """

        def extender(node, label, explored):
            result = []
            for a in self._domain.get_applicable_actions(node).get_elements():
                n = self._domain.get_next_state(node, a)
                if n not in explored:
                    result.append(
                        (
                            n,
                            self._domain.get_transition_value(node, a, n).cost,
                            {"action": a},
                        )
                    )
            return result

        push = heappush
        pop = heappop
        sources = [memory]
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
        enqueued = {
            source: (0, self._weight * self._heuristic(self._domain, source).cost)
            for source in sources
        }
        # enqueued = {source: min([(0, self._weight * self._heuristic(source, target, initial_label[source]).cost)
        # for target in targets], key=lambda x: x[1]) for source in sources}
        self.queue = [
            (enqueued[source][1], next(c), source, 0, None, initial_label[source])
            for source in sources
        ]
        # The explored dict is the CLOSED list.
        # It maps explored nodes to a pair of parent closest to the source and label of transition from parent.
        self.explored = {}
        path = []
        estim_total = 0.0
        while self.queue and not self.callback(self):
            # Pop the smallest item from queue, i.e. with smallest f-value
            estim_total, __, curnode, dist, parent, label = pop(self.queue)

            if self._render:
                self._domain.render(curnode)
            if self._verbose:
                print(
                    curnode,
                    f"- cumulated cost: {dist} - estimated total cost: {estim_total}",
                )
            if self._domain.is_goal(curnode):
                path = [(parent, label), (curnode, None)]
                node = parent
                while node is not None:
                    (parent, label) = self.explored[node]
                    if parent is not None:
                        path.insert(0, (parent, label))
                    node = parent
                break  # return path, dist, enqueued[curnode][0], len(enqueued)
            if curnode in self.explored:
                continue
            self.explored[curnode] = (parent, label)
            for neighbor, cost, lbl in extender(curnode, label, self.explored):
                if neighbor in self.explored:
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
                    # h = min([self._heuristic(neighbor, target, lbl).cost for target in targets])
                    h = self._heuristic(self._domain, neighbor).cost
                enqueued[neighbor] = ncost, h
                push(
                    self.queue,
                    (
                        ncost + (self._weight * h),
                        next(c),
                        neighbor,
                        ncost,
                        curnode,
                        lbl,
                    ),
                )
        self._policy: Dict[D.T_observation, Optional[D.T_event]] = {}
        for node, label in path:
            self._policy[node] = label["action"] if label is not None else None
            self._values[node] = estim_total - enqueued[node][0]
            if self._policy[node] is not None:
                self._plan.append(self._policy[node])
        # return estim_total, path  # TODO: find a way to expose these things through public API?

    def _get_next_action(
        self, observation: D.T_agent[D.T_observation]
    ) -> D.T_agent[D.T_concurrency[D.T_event]]:
        return self._policy[observation]

    def _is_policy_defined_for(self, observation: D.T_agent[D.T_observation]) -> bool:
        return observation in self._policy

    def _get_utility(self, observation: D.T_agent[D.T_observation]) -> D.T_value:
        if observation not in self._values:
            return self._heuristic(self._domain, observation).cost
        return self._values[observation]
