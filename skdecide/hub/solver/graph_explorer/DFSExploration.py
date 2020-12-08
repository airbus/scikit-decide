from skdecide.hub.solver.graph_explorer.GraphDomain import GraphDomain
from skdecide.hub.solver.graph_explorer.GraphExploration import GraphExploration


from skdecide import DeterministicPlanningDomain, Memory
from typing import Any


class DFSExploration(GraphExploration):
    def __init__(self, domain: DeterministicPlanningDomain,
                 max_edges=None,
                 max_nodes=None,
                 max_path=None):
        self.domain = domain
        self.max_edges = max_edges
        self.max_nodes = max_nodes
        self.max_path = max_path

    def build_graph_domain(self, init_state: Any=None,
                           transition_extractor=None,
                           verbose=True) -> GraphDomain:
        if transition_extractor is None:
            transition_extractor = lambda s, a, s_prime: {"cost":
                                                          self.domain.get_transition_value(s, a, s_prime)
                                                              .cost}
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
        paths_dict = {}
        while stack:
            (vertex, path) = stack.pop()
            actions = self.domain.get_applicable_actions(vertex).get_elements()
            for action in actions:
                next = self.domain.get_next_state(Memory([vertex]), action)
                if action not in next_state_map[vertex]:
                    nb_edges += 1
                else:
                    continue
                next_state_map[vertex][action] = next
                next_state_attributes[vertex][action] = transition_extractor(vertex, action, next)
                if self.domain.is_goal(next):
                    nb_path += 1
                    if verbose:
                        print(nb_path, " / ", self.max_path)
                        print("nodes  ", nb_nodes, " / ", self.max_nodes)
                        print("edges  ", nb_edges, " / ", self.max_edges)
                else:
                    if next not in next_state_map:
                        stack.append((next, path+[next]))
                        paths_dict[next] = set(tuple(path + [next]))
                    #else:
                    #     if tuple(path+[next]) not in paths_dict[next]:
                    #        stack.append((next, path + [next]))
                    #        paths_dict[next].add(tuple(path + [next]))
                if next not in next_state_map:
                    next_state_map[next] = {}
                    next_state_attributes[next] = {}
                    nb_nodes += 1
            if nb_path > self.max_path or (nb_nodes > self.max_nodes and nb_path >= 1) \
               or (nb_edges > self.max_edges and nb_path >= 1):
                break
        return GraphDomain(next_state_map,
                           next_state_attributes,
                           None, None)


