from skdecide.hub.solver.graph_explorer.GraphDomain import GraphDomain, GraphDomainUncertain
from skdecide.hub.solver.graph_explorer.GraphExploration import GraphExploration
from skdecide import DeterministicPlanningDomain, Memory, MDPDomain, GoalMDPDomain, D
from typing import Any, Dict, Tuple
from heapq import heappop, heappush
from itertools import count

# WARNING : adapted for the scheduling domains.


class DFSExploration(GraphExploration):
    def __init__(self, domain: GoalMDPDomain,
                 score_function=None,
                 max_edges=None,
                 max_nodes=None,
                 max_path=None):
        self.domain = domain
        self.score_function = score_function
        self.c = count()
        if score_function is None:
            self.score_function = lambda s: (next(self.c))
        self.max_edges = max_edges
        self.max_nodes = max_nodes
        self.max_path = max_path

    def build_graph_domain(self, init_state: Any = None) -> GraphDomainUncertain:
        if init_state is None:
            initial_state = self.domain.get_initial_state()
        else:
            initial_state = init_state
        stack = [(self.score_function(initial_state),
                  initial_state)]
        domain = self.domain
        goal_states = set()
        terminal_states = set()
        num_s = 0
        state_to_ind = {}
        nb_states = 1
        nb_edges = 0
        result = {initial_state}
        next_state_map: Dict[D.T_state, Dict[D.T_event, Dict[D.T_state, Tuple[float, float]]]] = {}
        state_terminal: Dict[D.T_state, bool] = {}
        state_goal: Dict[D.T_state, bool] = {}
        state_terminal[initial_state] = self.domain.is_terminal(initial_state)
        state_goal[initial_state] = self.domain.is_goal(initial_state)
        while len(stack) > 0:
            if not len(result) % 100 and len(result) > nb_states:
                print("Expanded {} states.".format(len(result)))
                nb_states = len(result)
            tuple, s = heappop(stack)
            if s not in state_to_ind:
                state_to_ind[s] = num_s
                num_s += 1
            if domain.is_terminal(s):
                terminal_states.add(s)
            if domain.is_goal(s):
                goal_states.add(s)
            if domain.is_goal(s) or domain.is_terminal(s):
                continue
            actions = domain.get_applicable_actions(s).get_elements()
            for action in actions:
                successors = domain.get_next_state_distribution(s, action).get_values()
                for succ, prob in successors:
                    if s not in next_state_map:
                        next_state_map[s] = {}
                    if action not in next_state_map[s]:
                        next_state_map[s][action] = {}
                    if prob != 0 and succ not in result:
                        nb_states += 1
                        nb_edges += 1
                        result.add(succ)
                        heappush(stack, (self.score_function(succ),
                                         succ))
                        cost = domain.get_transition_value(s, action, succ)
                        next_state_map[s][action][succ] = (prob, cost.cost)
                        state_goal[succ] = domain.is_goal(succ)
                        state_terminal[succ] = domain.is_terminal(succ)
            if (nb_states > self.max_nodes) \
                or (nb_edges > self.max_edges):
                break
        return GraphDomainUncertain(next_state_map=next_state_map,
                                    state_terminal=state_terminal,
                                    state_goal=state_goal)

