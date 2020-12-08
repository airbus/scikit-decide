import math
from typing import List
from collections import namedtuple
from skdecide.builders.discrete_optimization.vrp.vrp_model import BasicCustomer, VrpProblem, VrpSolution
import networkx as nx
import numpy as np

Customer = namedtuple("Customer", ['index', 'demand', 'x', 'y'])


def compute_length_matrix(vrp_model: VrpProblem):
    nb_customers = vrp_model.customer_count
    matrix_distance = np.zeros((nb_customers, nb_customers))
    for f in range(nb_customers):
        for c in range(f+1, nb_customers):
            matrix_distance[f, c] = vrp_model.evaluate_function_indexes(f, c)
            matrix_distance[c, f] = matrix_distance[f, c]
    closest = np.argsort(matrix_distance, axis=1)
    return closest, matrix_distance


def prune_search_space(vrp_model: VrpProblem,
                       n_shortest=10):
    closest, matrix_distance = compute_length_matrix(vrp_model)
    matrix_adjacency = np.zeros(matrix_distance.shape, dtype=np.int)
    nb_customers = vrp_model.customer_count
    if n_shortest < nb_customers:
        for c in range(matrix_adjacency.shape[0]):
            matrix_adjacency[c, closest[c, :n_shortest]] = matrix_distance[c, closest[c, :n_shortest]]
            matrix_adjacency[c, 0] = matrix_distance[c, 0]
    else:
        matrix_adjacency = matrix_distance
    return matrix_adjacency, matrix_distance


def build_graph(vrp_model: VrpProblem):
    matrix_adjacency, matrix_distance = prune_search_space(vrp_model=vrp_model,
                                                           n_shortest=vrp_model.customer_count)
    G = nx.from_numpy_matrix(matrix_adjacency, create_using=nx.DiGraph)
    G.add_edge(0, 0, weight=0)
    return G, matrix_distance


def length(customer1, customer2):
    return math.sqrt((customer1.x - customer2.x)**2 + (customer1.y - customer2.y)**2)


def parse_input(input_data):
    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    customer_count = int(parts[0])
    vehicle_count = int(parts[1])
    vehicle_capacity = int(parts[2])

    customers = []
    for i in range(1, customer_count+1):
        line = lines[i]
        parts = line.split()
        customers.append(Customer(i-1, int(parts[0]), float(parts[1]), float(parts[2])))
    return vehicle_count, vehicle_capacity, customer_count, customers


def recompute_output(dictionary_path, customers):
    vehicle_tours = []
    obj = 0
    for k in dictionary_path:
        path = dictionary_path[k]+[0]
        range_index = range(len(path))
        obj += sum([length(customers[ii], customers[jj]) for ii, jj in zip(path[:-1], path[1:])])
        vehicle_tours += [[customers[j] for j in path[1:-1]]]
    return vehicle_tours, obj, 0


def output_result(vehicle_tours, obj, opt, vehicle_count, depot):
    # prepare the solution in the specified output format
    outputData = '%.2f' % obj + ' ' + str(opt) + '\n'
    print(len(vehicle_tours))
    for v in range(0, vehicle_count):
        outputData += str(depot.index) + ' ' + ' '.join([str(customer.index) for customer in vehicle_tours[v]]) + ' ' + str(depot.index) + '\n'
    return outputData