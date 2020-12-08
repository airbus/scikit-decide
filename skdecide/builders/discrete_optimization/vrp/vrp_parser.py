import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
import os
from skdecide.builders.discrete_optimization.generic_tools.graph_api import Graph
from skdecide.builders.discrete_optimization.vrp.vrp_model import VrpProblem, BasicCustomer, Customer2D, VrpProblem2D
path_to_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/vrp/")
files_available = [os.path.join(path_to_data, f) for f in os.listdir(path_to_data)]

from collections import namedtuple
from typing import Union


def parse_input(input_data, start_index=0, end_index=0, vehicle_count=None):
    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    customer_count = int(parts[0])
    vehicle_count = int(parts[1]) if vehicle_count is None else vehicle_count
    vehicle_capacity = int(parts[2])
    customers = []
    for i in range(1, customer_count+1):
        line = lines[i]
        parts = line.split()
        customers.append(Customer2D(i-1, int(parts[0]),
                                    float(parts[1]), float(parts[2])))
    vehicle_capacities = [vehicle_capacity]*vehicle_count
    start_indexes = [start_index]*vehicle_count
    end_indexes = [end_index]*vehicle_count
    return VrpProblem2D(vehicle_count=vehicle_count,
                        vehicle_capacities=vehicle_capacities,
                        customer_count=customer_count,
                        customers=customers,
                        start_indexes=start_indexes,
                        end_indexes=end_indexes)


def parse_file(file_path, start_index=0, end_index=0, vehicle_count=None)->VrpProblem:
    with open(file_path, 'r') as input_data_file:
        input_data = input_data_file.read()
        vrp_model = parse_input(input_data, start_index=start_index, end_index=end_index, vehicle_count=vehicle_count)
        return vrp_model


