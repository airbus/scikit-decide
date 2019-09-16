# TODO: update to new API

from typing import Optional, Callable, Any, Iterable

from airlaps import Memory, T_observation, T_event, Domain
from airlaps.builders.domain import EnumerableTransitionDomain, ActionDomain, GoalDomain, \
    DeterministicInitializedDomain, MarkovianDomain, PositiveCostDomain
from airlaps.builders.solver import DomainSolver, DeterministicPolicySolver, SolutionSolver, UtilitySolver


class LRTAstar(DomainSolver, DeterministicPolicySolver, SolutionSolver, UtilitySolver):

    def __init__(self, heuristic: Optional[Callable[[T_observation, Domain], float]] = None,
                 weight: float = 1.) -> None:
        self._heuristic = (lambda _, __: 0.) if heuristic is None else heuristic
        self._weight = weight
        self.max_iter = 5000
        self.max_depth = 200
        self._plan = []
        self.values = {}

        self.heuristic_changed = False
        self._policy = {}

    def _reset(self) -> None:
        self._domain = self._new_domain()
        self.values = {}

        def extender(node, label, explored):
            memory = Memory([node])
            neigh = [(self._domain.get_next_state(memory, a), a)
                     for a in self._domain.get_applicable_actions(memory).get_elements()]
            neigh_not_explored = [(n, a) for n, a in neigh if n not in explored]
            cost = {n: self._domain.get_transition_value(memory, a, n).cost for n, a in neigh_not_explored}
            labels = {n: {'action': a} for n, a in neigh_not_explored}
            return {n: (cost[n], labels[n]) for n, _ in neigh_not_explored}

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
        pass  # TODO: check that the goal space is an EnumerableSpace (using "from inspect import signature")

    def get_next_action(self, memory: Memory[T_observation]) -> T_event:
        current_state = memory[-1]
        if current_state not in self._policy:
            self.solve(from_observation=memory)
        return self._policy[current_state]

    def is_policy_defined_for(self, memory: Memory[T_observation]) -> bool:
        current_state = memory[-1]
        return current_state in self._policy

    def solve(self, from_observation: Optional[Memory[T_observation]] = None,
              on_update: Optional[Callable[..., bool]] = None,
              max_time: Optional[float] = None, **kwargs: Any):
        self.values = {}
        iteration = 0
        best_cost = float('inf')
        best_path = None
        while True:
            dead_end, cumulated_cost, current_roll, list_action = self.doTrial(from_observation)
            if not dead_end and cumulated_cost < best_cost:
                best_cost = cumulated_cost
                best_path = current_roll
                for k in range(len(current_roll)):
                    self._policy[current_roll[k][0]] = current_roll[k][1]["action"]
            if not self.heuristic_changed:
                return best_cost, best_path
            iteration += 1
            if iteration > self.max_iter:
                return best_cost, best_path

    def doTrial(self, from_observation: Optional[Memory[T_observation]]):
        list_action = []
        current_state = from_observation[-1]
        depth = 0
        dead_end = False
        cumulated_reward = 0.
        current_roll = [current_state]
        current_roll_and_action = []
        self.heuristic_changed = False
        while (not self._domain.is_goal(current_state)) and (depth < self.max_depth):
            next_action = None
            next_state = None
            best_estimated_cost = float('inf')
            applicable_actions = self._domain.get_applicable_actions(Memory([current_state]))
            for action in applicable_actions.get_elements():
                st = self._domain.get_next_state(Memory([current_state]), action)
                r = self._domain.get_transition_value(Memory([current_state]), action, st).cost
                if st in current_roll:
                    continue
                if st not in self.values:
                    self.values[st] = self._heuristic(st, self._domain)
                if r + self.values[st] < best_estimated_cost:
                    next_state = st
                    next_action = action
                    best_estimated_cost = r + self.values[st]
            if next_action is None:
                self.values[current_state] = float('inf')
                dead_end = True
                self.heuristic_changed = True
                break
            else:
                if (not current_state in self.values) or (self.values[current_state] != best_estimated_cost):
                    self.heuristic_changed = True
                    self.values[current_state] = best_estimated_cost
            cumulated_reward += best_estimated_cost - (self.values[next_state] if next_state in self.values else
                                                       self._heuristic(next_state, self._domain))
            list_action.append(next_action)
            current_roll_and_action.append((current_state, {"action": next_action}))
            current_state = next_state
            depth += 1
            current_roll.append(current_state)
        current_roll_and_action.append((current_state, {"action": None}))
        cumulated_reward += self.values[current_state]
        return dead_end, cumulated_reward, current_roll_and_action, list_action
