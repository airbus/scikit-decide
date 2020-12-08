
from skdecide.hub.solver.graph_explorer.GraphDomain import GraphDomain
from skdecide.hub.solver.graph_explorer.GraphExploration import GraphExploration


from skdecide import DeterministicPlanningDomain, Memory
from typing import Any


class FullSpaceExploration(GraphExploration):
    def __init__(self, domain: DeterministicPlanningDomain,
                 max_edges=None,
                 max_nodes=None,
                 max_path=None):
        self.domain = domain
        self.max_edges = max_edges
        self.max_nodes = max_nodes
        self.max_path = max_path

    def build_graph_domain(self, init_state: Any=None) -> GraphDomain:
        next_state_map = {}
        next_state_attributes = {}
        if init_state is None:
            init_state = self.domain.get_initial_state()
        stack = [(init_state, [init_state])]
        nb_nodes = 1
        nb_edges = 0
        nb_path = 0
        next_state_map[init_state] = {}
        next_state_attributes[init_state] = {}
        while stack:
            (vertex, path) = stack.pop()
            actions = self.domain.get_applicable_actions(vertex).get_elements()
            for action in actions:
                next = self.domain.get_next_state(vertex, action)
                if next not in next_state_map:
                    next_state_map[next] = {}
                    next_state_attributes[next] = {}
                    nb_nodes += 1
                if action not in next_state_map[vertex]:
                    nb_edges += 1
                next_state_map[vertex][action] = next
                next_state_attributes[vertex][action] = {"cost": self.domain.get_transition_value(Memory([vertex]),
                                                                                                  action,
                                                                                                  next).cost,
                                                         "reward": self.domain.get_transition_value(Memory([vertex]),
                                                                                                    action,
                                                                                                    next).reward}
                if self.domain.is_goal(next):
                    nb_path += 1
                else:
                    if next not in next_state_map:
                        stack.append((next, path+[next]))
            if nb_path > self.max_path or (nb_nodes > self.max_nodes and nb_path >= 1) \
                or (nb_edges > self.max_edges and nb_path >= 1):
                break
        return GraphDomain(next_state_map,
                           next_state_attributes,
                           None, None)


def reachable_states(self, s0: Any):
    """ Computes all states reachable from s0.
    """
    result = {s0}
    stack = [s0]
    domain = self._domain
    while len(stack) > 0:
        if not len(result) % 100:
            print("Expanded {} states.".format(len(result)))
        s = stack.pop()
        if domain.is_terminal(s):
            continue
        # Add successors
        actions = domain.get_applicable_actions(s).get_elements()
        for action in actions:
            successors = domain.get_next_state_distribution(s, action).get_values()
            for succ, prob in successors:
                if prob != 0 and succ not in result:
                    result.add(succ)
                    stack.append(succ)
    return result