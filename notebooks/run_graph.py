import argparse
import logging
import networkx as nx
import numpy as np
import random
import time
import tqdm
from typing import Sequence

from graph_domain import GraphDomain, GraphState
from skdecide.hub.solver.lazy_astar import LazyAstar


def parse_gr(file: str = None):
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


def run(file: str, repeat: int, seed: int, algos: Sequence[str]) -> np.array:

    graph, dict_graph = parse_gr(file)
    logging.info("Graph loaded")

    list_nodes = list(dict_graph.keys())
    graph_domain = GraphDomain(next_state_map={n1: {n2: n2 for n2 in dict_graph[n1]}
                                               for n1 in dict_graph},
                               next_state_attributes={n1: {n2: {"weight": dict_graph[n1][n2]}
                                                           for n2 in dict_graph[n1]}
                                                      for n1 in dict_graph},
                               targets=None, attribute_weight="weight")

    if seed >= 0:
        random.seed(a=seed)

    timings = np.zeros((repeat, 3), dtype=float)
    print("repeat=", repeat)
    for k in range(repeat):
        dep = random.choice(list_nodes)
        arr = random.choice(list_nodes)

        logging.info(f"Starting from {dep} to {arr} ({k+1}/{repeat})")

        if algos is None or "networkx" in algos:
            name = "astar networkx"
            t = time.time()
            path = nx.astar_path(graph, dep, arr, weight="weight")
            timings[k, 1] = time.time()-t
            timings[k, 0] = sum(graph[n1][n2]["weight"] for n1, n2 in zip(path[:-1], path[1:]))
            logging.info(f"time: {timings[k, 1]}, {name} - length : {timings[k, 0]}")

        if algos is None or "astar" in algos:
            name = "scikit-decide"
            t = time.time()
            l = LazyAstar(from_state=GraphState(dep))
            graph_domain.targets = {arr}
            l.solve(domain_factory=lambda: graph_domain)
            timings[k, 2] = time.time()-t
            timings[k, 0] = l._values[GraphState(dep)]
            logging.info(f"time: {timings[k, 2]}, {name} - length : {timings[k, 0]}")

    return timings

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Solve Graph problem using various algorithms")
    parser.add_argument("--graph_file", type=str, required=True, help="SM file")
    parser.add_argument("--repeat", type=int, required=False, default=50, help="number of runs")
    parser.add_argument("--seed", type=int, required=False, default=-1, help="random seed (by default, seed is not fixed)")
    parser.add_argument("--output", type=str, required=False, help="CSV file containing timing results")
    parser.add_argument("--algo", type=str, action="append", required=False, choices=["networkx", "astar"], help="algorithm to run (can be repeated)")
    args = parser.parse_args()

    results = run(file=args.graph_file, repeat=args.repeat, seed=args.seed, algos=args.algo)
    if args.output:
        np.savetxt(args.output, results, fmt="%8.3e")
