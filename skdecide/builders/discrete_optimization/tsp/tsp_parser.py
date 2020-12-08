import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../"))
from skdecide.builders.discrete_optimization.tsp.tsp_model import TSPModel2D, Point2D
this_folder = os.path.dirname(os.path.abspath(__file__))
folder_data = os.path.join(this_folder, "../data/tsp/")


def get_data_available():
    files = [f for f in os.listdir(folder_data) if "pk" not in f and "json" not in f]
    return [os.path.join(folder_data, f) for f in files]


def parse_file(file_path, start_index=None, end_index=None):
    # parse the input
    with open(file_path, 'r') as input_data_file:
        input_data = input_data_file.read()
        lines = input_data.split('\n')
        node_count = int(lines[0])
        points = []
        for i in range(1, node_count+1):
            line = lines[i]
            parts = line.split()
            points.append(Point2D(float(parts[0]), float(parts[1])))
        return TSPModel2D(list_points=points,
                          node_count=node_count, 
                          start_index=start_index,
                          end_index=end_index,
                          use_numba=False)