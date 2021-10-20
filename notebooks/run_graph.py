import argparse
import logging
import networkx as nx
import random
import time
import tqdm

from graph_domain import GraphDomain
from skdecide.hub.solver.lazy_astar import LazyAstar

from stateful_graph import GraphState, GraphDomainStateful


def parse_gr(file:str = None):
    with open(file, "r") as input_data:
        lines = input_data.read().split('\n')
        graph = nx.Graph()
        dict_graph = {}
        # Retrieving section bounds
        for j in tqdm.tqdm(range(len(lines))):
            split = lines[j].split()
            if len(split)==0:
                continue
            if split[0] == "a":
                node1 = int(split[1])
                node2 = int(split[2])
                distance = float(split[3])
                if node1 not in graph:
                    graph.add_node(node1)
                    dict_graph[node1] = {}
                if node2 not in graph:
                    graph.add_node(node2)
                    dict_graph[node2] = {}
                graph.add_edge(node1, node2, weight=distance)
                dict_graph[node1][node2] = distance
                dict_graph[node2][node1] = distance
    return graph, dict_graph


def run(file:str = None):

    assert file != None, "No file provided"

    graph, dict_graph = parse_gr(file)
    logging.info("Graph loaded")

    list_nodes = list(dict_graph.keys())
    graph_domain = GraphDomain(next_state_map={n1: {n2: n2 for n2 in dict_graph[n1]}
                                               for n1 in dict_graph},
                               next_state_attributes={n1: {n2: {"weight": dict_graph[n1][n2]}
                                                           for n2 in dict_graph[n1]}
                                                      for n1 in dict_graph},
                               targets=None, attribute_weight="weight")

    stateful_graph_domain = GraphDomainStateful(next_state_map={n1: {n2: n2 for n2 in dict_graph[n1]}
                                                          for n1 in dict_graph},
                                          next_state_attributes={n1: {n2: {"weight": dict_graph[n1][n2]}
                                                                      for n2 in dict_graph[n1]}
                                                                 for n1 in dict_graph},
                                          targets=None,
                                          attribute_weight="weight")

    for k in range(50):
        dep = random.choice(list_nodes)
        arr = random.choice(list_nodes)

        logging.info("Starting from {} to {}".format(dep, arr))

        t = time.time()
        path = nx.astar_path(graph, dep, arr, weight="weight")
        length = sum(graph[n1][n2]["weight"] for n1, n2 in zip(path[:-1], path[1:]))
        logging.info(f"time: {time.time()-t}, astar networkx - length : {length}")

        t = time.time()
        l = LazyAstar(from_state=dep)
        graph_domain.targets = {arr}
        l.solve(domain_factory=lambda: graph_domain)
        logging.info(f"time: {time.time()-t}, scikit-decide - length : {l._values[dep]}")

        t = time.time()
        l = LazyAstar(from_state=GraphState(dep))
        stateful_graph_domain.targets = {arr}
        l.solve(domain_factory=lambda: stateful_graph_domain)
        logging.info(f"time: {time.time()-t}, scikit-decide stateful - length : {l._values[GraphState(dep)]}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Solve Graph problem using various algorithms")
    parser.add_argument("--graph_file", type=str, required=True, help="SM file")
    args = parser.parse_args()

    run(args.graph_file)
