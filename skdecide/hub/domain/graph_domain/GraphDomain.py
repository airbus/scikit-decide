# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import networkx as nx

from skdecide import (
    D,
    DeterministicPlanningDomain,
    Distribution,
    Domain,
    EnumerableSpace,
    ImplicitSpace,
    Memory,
    SamplableSpace,
    Space,
    T,
    Value,
)
from skdecide.builders.domain import (
    Actions,
    DeterministicTransitions,
    FullyObservable,
    Goals,
    Markovian,
    PositiveCosts,
    UncertainTransitions,
)


class ActionSpace(EnumerableSpace, SamplableSpace):
    def sample(self) -> T:
        return random.choice(self.l)

    def contains(self, x: T) -> bool:
        pass

    def __init__(self, l: List[object]):
        self.l = l

    def get_elements(self) -> Iterable[object]:
        return self.l


class GraphDomainUncertain(
    Domain,
    UncertainTransitions,
    Actions,
    Goals,
    Markovian,
    FullyObservable,
    PositiveCosts,
):
    """
    This domain is for uncertain goal MDP where the full transitions, probabilities and cost are already computed.
    In this case, using the dictionary structures will improve the computing performance of the domain and therefore
    its solving time
    """

    def __init__(
        self,
        next_state_map: Dict[
            D.T_state, Dict[D.T_event, Dict[D.T_state, Tuple[float, float]]]
        ],
        state_terminal: Dict[D.T_state, bool],
        state_goal: Dict[D.T_state, bool],
    ):
        """
        # Parameters
        - next_state_map : a dictionary whose keys are state and values are dictionary with actions as keys
            and as value another dict with next state as keys and with (proba, cost) as value.
            This format could be changed in the future.
        - state_terminal: a dictionary indicating for each state if it's terminal or not
        - state_goal: a dictionary indicating for each state if it's a goal or not
        """
        self.next_state_map = next_state_map  # State, action, -> next state
        self.state_terminal = state_terminal
        self.state_goal = state_goal

    def to_networkx(self):
        graph = nx.DiGraph()
        states_list = list(self.state_terminal.keys())
        id_to_state = {i: states_list[i] for i in range(len(states_list))}
        state_to_id = {states_list[i]: i for i in range(len(states_list))}
        for i in id_to_state:
            graph.add_node(i, state=id_to_state[i])
        for state in self.next_state_map:
            for action in self.next_state_map[state]:
                for next_state in self.next_state_map[state][action]:
                    graph.add_edge(
                        state_to_id[state],
                        state_to_id[next_state],
                        action=action,
                        proba=self.next_state_map[state][action][next_state][0],
                        cost=self.next_state_map[state][action][next_state][1],
                    )
        return graph

    def _get_transition_value(
        self,
        memory: D.T_memory[D.T_state],
        event: D.T_event,
        next_state: Optional[D.T_state] = None,
    ) -> Value[D.T_value]:
        return Value(cost=self.next_state_map[memory][event][next_state][1])

    def _is_terminal(self, state: D.T_state) -> bool:
        return (
            self.state_terminal[state]
            or len(self._get_applicable_actions_from(state).l) == 0
        )

    def _get_action_space_(self) -> Space[D.T_event]:
        return ImplicitSpace(lambda x: True)

    def _get_applicable_actions_from(
        self, memory: D.T_memory[D.T_state]
    ) -> Space[D.T_event]:
        return ActionSpace(list(self.next_state_map.get(memory, {}).keys()))

    def _get_goals_(self) -> Space[D.T_observation]:
        return ImplicitSpace(lambda x: self.state_goal[x])

    def _is_goal(self, state: D.T_state) -> bool:
        return self.state_goal[state]

    def _get_next_state_distribution(
        self,
        memory: D.T_memory[D.T_state],
        action: D.T_agent[D.T_concurrency[D.T_event]],
    ) -> Distribution[D.T_state]:
        possible_states = [
            (s, self.next_state_map[memory][action][s][0])
            for s in self.next_state_map[memory][action]
        ]
        return random.choices(
            possible_states, weights=[p[1] for p in possible_states], k=1
        )[0]


class GraphDomain(DeterministicPlanningDomain):
    """
    This domain is for deterministic planning domain where the full transitions and cost are already computed.
    In this case, using the dictionary structures will improve the computing performance of the domain and therefore
    its solving time.
    """

    def __init__(
        self,
        next_state_map: Dict[D.T_state, Dict[D.T_event, D.T_state]],
        next_state_attributes: Dict[D.T_state, Dict[D.T_event, Dict[str, float]]],
        targets: Optional[Set[D.T_state]] = None,
        attribute_weight="weight",
    ):
        """

        # Parameters
        - next_state_map: is a dictionary with keys the state and values a dictionary
        with action as keys and next state as values
        - next_state_attributes: for each transition, stores float attributes (typically cost of transition)
        - target : set of goal states
        - attribute_weight: key in next_state_attributes to consider as the cost attribute.
        """
        self.next_state_map = next_state_map  # State, action, -> next state
        self.next_state_attributes = next_state_attributes
        if targets is None:
            self.targets = set()
        else:
            self.targets = set(targets)
        self.attribute_weight = attribute_weight

    def merge(self, graph_domain: GraphDomain):
        """Return a new graph domain merged from self and another instance of GraphDomain."""
        next_state_map = self.next_state_map
        next_state_attributes = self.next_state_attributes
        for k in graph_domain.next_state_map:
            if k not in next_state_map:
                next_state_map[k] = graph_domain.next_state_map[k]
                next_state_attributes[k] = graph_domain.next_state_attributes[k]
            else:
                for action in graph_domain.next_state_map[k]:
                    if action not in next_state_map[k]:
                        next_state_map[k][action] = graph_domain.next_state_map[k][
                            action
                        ]
                        next_state_attributes[k][
                            action
                        ] = graph_domain.next_state_attributes[k][action]
        return GraphDomain(
            next_state_map, next_state_attributes, self.targets, self.attribute_weight
        )

    def _get_next_state(
        self, memory: D.T_memory[D.T_state], event: D.T_event
    ) -> D.T_state:
        return self.next_state_map[memory][event]

    def _get_transition_value(
        self,
        memory: D.T_memory[D.T_state],
        event: D.T_event,
        next_state: Optional[D.T_state] = None,
    ) -> Value[D.T_value]:
        return Value(
            cost=self.next_state_attributes[memory][event][self.attribute_weight]
        )

    def is_terminal(self, state: D.T_state) -> bool:
        return state in self.targets

    def _get_action_space_(self) -> Space[D.T_event]:
        return ImplicitSpace(lambda x: True)

    def _get_applicable_actions_from(self, memory: D.T_state) -> Space[D.T_event]:
        return ActionSpace(list(self.next_state_map[memory].keys()))

    def _get_goals_(self) -> Space[D.T_observation]:
        return ImplicitSpace(lambda x: (x in self.targets))

    def is_goal(self, state: D.T_state) -> bool:
        return state in self.targets

    def _get_observation_space_(self) -> Space[D.T_observation]:
        pass

    def set_sources_targets(self, sources, targets):
        """
        Change the sources and targets attribute.
        """
        self.sources = sources
        self.targets = targets

    def set_nodes_target(self, targets):
        """
        Change the sources and targets attribute.
        """
        self.targets = targets
