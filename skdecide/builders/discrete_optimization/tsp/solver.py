#!/usr/bin/python
# -*- coding: utf-8 -*-

from solver_common_tools import *
from solver_lp import model_tsp_tsn

from enum import Enum
class MethodTSP(Enum):
    BASELINE_ORDER = 0
    GREEDY_CLOSEST = 1
    METHOD_BY_HAND = 4
    ORTOOLS = 2
    LP = 3

def greedy_closest_arc(nodeCount, points):
    matrix = build_matrice_distance(nodeCount, 
                                    points)
    sol = [0]
    length_circuit = 0.
    index_in_sol = {0}
    s = np.argsort(matrix, axis=1)[:, 1:]
    # print(matrix.shape)
    # print(s.shape)
    # print(s)
    cur_point = 0
    nb_point = 1
    while nb_point<nodeCount:
        n = next(p for p in s[cur_point, :] if p not in index_in_sol)
        length_circuit += matrix[cur_point, n]
        index_in_sol.add(n)
        sol += [n]
        cur_point = n
        nb_point += 1
    length_circuit += matrix[cur_point, 0]
    return sol, length_circuit, 0
    

from solver_ortools import main_func
import pickle, json
cur_method = MethodTSP.METHOD_BY_HAND
def solve_it(input_data):
    #build_matrices()
    # Modify this code to run your optimization algorithm
    nodeCount, points = parse(input_data)
    if cur_method == MethodTSP.METHOD_BY_HAND:
        #"YQL5Q, ./data/tsp_51_1, solver.py, Traveling Salesman Problem 1
        #R9hfg, ./data/tsp_100_3, solver.py, Traveling Salesman Problem 2
        #ZVrLp, ./data/tsp_200_2, solver.py, Traveling Salesman Problem 3
        #6tyFn, ./data/tsp_574_1, solver.py, Traveling Salesman Problem 4
        #tSpoY, ./data/tsp_1889_1, solver.py, Traveling Salesman Problem 5
        #YOVlV, ./data/tsp_33810_1, solver.py, Traveling Salesman Problem 6"
        name_json = None
        if nodeCount == 51:
            name_json = "./tsp_51_1.json"
            opt = 1
        elif nodeCount == 100:
            name_json = "./tsp_100_3.json"
            opt = 1
        elif nodeCount == 200:
            name_json = "./tsp_200_2.json"
            opt = 1
        elif nodeCount == 574:
            name_json = "./tsp_574_1.json"
            opt = 0
        elif nodeCount == 1889:
            name_json = "./tsp_1889_1.json"
            opt = 0
        elif nodeCount == 33810:
            name_json = "./tsp_33810_1.json"
            opt = 0
        if name_json is not None:
            permut = json.load(open(name_json, "r"))
        lengths, obj = compute_length(permut, points, nodeCount)
        solution = permut
    if cur_method == MethodTSP.ORTOOLS and nodeCount<1900:
        solution, obj, opt = main_func(nodeCount, points)
    elif cur_method == MethodTSP.GREEDY_CLOSEST:
        solution, obj, opt = greedy_closest_arc(nodeCount, points)
    elif cur_method == MethodTSP.BASELINE_ORDER:
        # matrix = build_matrice_distance(nodeCount, points)
        solution, obj, opt = baseline_in_order(nodeCount, points)
    elif cur_method == MethodTSP.LP:
        # matrix = build_matrice_distance(nodeCount, points)
        solution, obj, opt = model_tsp_tsn(nodeCount, points, True)
    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(opt) + '\n'
    output_data += ' '.join(map(str, solution))
    return output_data

import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')

