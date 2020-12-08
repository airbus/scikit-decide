import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../"))
from skdecide.builders.discrete_optimization.knapsack.knapsack_model import Item, KnapsackModel
import os
import sys
from skdecide.builders.discrete_optimization.generic_tools.path_tools import abspath_from_file
path_to_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/knapsack/")
files_available = [os.path.join(path_to_data, f) for f in os.listdir(path_to_data)]


def parse_input_data(input_data, force_recompute_values: bool=False)->KnapsackModel:
    """
    Parse a string of the following form :
    item_count max_capacity
    item1_value item1_weight
    ...
    itemN_value itemN_weight
    """
    lines = input_data.split('\n')
    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])
    items = []
    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))
    return KnapsackModel(list_items=items, max_capacity=capacity, force_recompute_values=force_recompute_values)


def parse_file(file_path, force_recompute_values=False)->KnapsackModel:
    with open(file_path, 'r') as input_data_file:
        input_data = input_data_file.read()
        knapsack_model = parse_input_data(input_data, force_recompute_values=force_recompute_values)
        return knapsack_model


def test_parser():
    file_location = os.path.join(abspath_from_file(__file__, "./data/ks_4_0"))
    knapsack_model = parse_file(file_location)
    print(knapsack_model)


if __name__ == '__main__':
    test_parser()
