import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
import os
from skdecide.builders.discrete_optimization.generic_tools.graph_api import Graph
from skdecide.builders.discrete_optimization.coloring.coloring_model import ColoringProblem
path_to_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/coloring/")
files_available = [os.path.join(path_to_data, f) for f in os.listdir(path_to_data)]


def parse(input_data):
    # parse the input
    lines = input_data.split('\n')
    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])
    edges = []
    nodes = [(i, {}) for i in range(node_count)]
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1]), {}))
    return ColoringProblem(Graph(nodes, edges, undirected=True))


def parse_file(file_path)->ColoringProblem:
    with open(file_path, 'r') as input_data_file:
        input_data = input_data_file.read()
        coloring_model = parse(input_data)
        return coloring_model


if __name__ == "__main__":
    file = files_available[0]
    model = parse_file(file)
    print(model.graph.nodes_name)
