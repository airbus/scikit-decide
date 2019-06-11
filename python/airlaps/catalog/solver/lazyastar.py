from heapq import heappush, heappop
from itertools import count
from typing import Optional, Callable, Any, Iterable

from airlaps import Memory, T_observation, T_event, Domain
from airlaps.builders.domain import EnumerableTransitionDomain, ActionDomain, GoalDomain, \
    DeterministicInitializedDomain, MarkovianDomain, PositiveCostDomain
from airlaps.builders.solver import DomainSolver, DeterministicPolicySolver, SolutionSolver, UtilitySolver


class LazyAstar(DomainSolver, DeterministicPolicySolver, SolutionSolver, UtilitySolver):

    def __init__(self, heuristic: Optional[Callable[[T_observation, Domain], float]] = None,
                 weight: float = 1.) -> None:
        self._heuristic = (lambda _, __: 0.) if heuristic is None else heuristic
        self._weight = weight
        self.values = {}
        self._plan = []

    def _reset(self) -> None:
        self._domain = self._new_domain()

        def extender(node, label, explored):
            memory = Memory([node])
            # memory = node  # works also but slightly less efficiently (since node is wrapped in new Memory every time)
            neigh = [(self._domain.get_next_state(memory, a), a)
                     for a in self._domain.get_applicable_actions(memory).get_elements()]
            neigh_not_explored = [(n, a) for n, a in neigh if n not in explored]
            cost_labels = [(n, self._domain.get_transition_value(memory, a, n).cost, {'action': a})
                           for n, a in neigh_not_explored]
            return cost_labels

        self._extender = extender

    def get_utility(self, memory: Memory[T_observation]) -> float:
        state = memory[-1]
        if state not in self.values:
            return self._heuristic(state, self._domain)
        return self.values[state]

    def get_domain_requirements(self) -> Iterable[type]:
        # TODO: relax constraint on "DeterministicInitializedDomain" (since the algo can have several sources) and
        #  "MarkovianDomain" (useful for technical reasons: memory is not hashable whereas we assume here that current
        #  state is)?
        return [EnumerableTransitionDomain, ActionDomain, GoalDomain, DeterministicInitializedDomain, MarkovianDomain,
                PositiveCostDomain]

    def _check_domain(self, domain: Domain) -> bool:
        return True  # TODO: check that the goal space is an EnumerableSpace (using "from inspect import signature")

    def get_next_action(self, memory: Memory[T_observation]) -> T_event:
        current_state = memory[-1]
        return self._policy[current_state]

    def is_policy_defined_for(self, memory: Memory[T_observation]) -> bool:
        current_state = memory[-1]
        return current_state in self._policy

    def solve(self, from_observation: Optional[Memory[T_observation]] = None,
              on_update: Optional[Callable[..., bool]] = None, max_time: Optional[float] = None,
              **kwargs: Any) -> tuple:
        assert from_observation is not None  # TODO: if None get internal domain memory?
        verbose = kwargs.get('verbose', False)
        render = kwargs.get('render', False)
        push = heappush
        pop = heappop
        sources = [from_observation[-1]]
        # targets = list(self._domain.get_goals().get_elements())

        # The queue is the OPEN list.
        # It stores priority, node, cost to reach, parent and label (any data type) of transition from parent.
        # Uses Python heapq to keep in priority order.
        # Add a counter to the queue to prevent the underlying heap from
        # attempting to compare the nodes themselves. The hash breaks ties in the
        # priority and is guaranteed unique for all nodes in the graph.
        c = count()

        initial_label = {source: None for source in
                         sources}  # TODO: check if necessary (a priori used to keep additional infos)
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
            if render:
                self._domain.render(curnode)
            if verbose:
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
            for neighbor, cost, lbl in self._extender(curnode, label, explored):
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
            self.values[node] = estim_total - enqueued[node][0]
            if self._policy[node] is not None:
                self._plan.append(self._policy[node])
        return estim_total, path

        # return [], 0, 0, len(enqueued)
