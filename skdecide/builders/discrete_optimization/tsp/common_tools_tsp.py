
import math
from collections import namedtuple
import os
import networkx as nx
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from skdecide.builders.discrete_optimization.tsp.tsp_model import Point2D, length
this_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(this_path)


def length_1(point1, point2):
    return abs(point1.x - point2.x) + abs(point1.y - point2.y)


def parse(input_data):
    # parse the input
    lines = input_data.split('\n')
    nodeCount = int(lines[0])
    points = []
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        points.append(Point2D(float(parts[0]), float(parts[1])))
    return nodeCount, points


def compute_length(solution, list_points, nodeCount):
    obj = length(list_points[solution[-1]], list_points[solution[0]])
    lengths = []
    pp = obj
    for index in range(0, nodeCount-1):
        ll = length(list_points[solution[index]], list_points[solution[index+1]])
        obj += ll
        lengths += [ll]
    lengths += [pp]
    return lengths, obj


def baseline_in_order(nodeCount: int, points: List[Point2D]):
    # build a trivial solution
    # visit the nodes in the order they appear in the file
    solution = range(0, nodeCount)

    # calculate the length of the tour
    obj = length(points[solution[-1]], points[solution[0]])
    for index in range(0, nodeCount-1):
        obj += length(points[solution[index]], points[solution[index+1]])
    return solution, obj, 0


def build_matrice_distance(nodeCount: int, points: List[Point2D], method=None):
    if method is None:
        method = length
    matrix = np.zeros((nodeCount, nodeCount))
    for i in range(nodeCount):
        for j in range(i+1, nodeCount):
            d = method(i, j)
            matrix[i, j] = d
            matrix[j, i] = d
    return matrix


def build_matrice_distance_np(nodeCount: int, points: List[Point2D]):
    matrix_x = np.ones((nodeCount, nodeCount), dtype=np.int32)
    matrix_y = np.ones((nodeCount, nodeCount), dtype=np.int32)
    print("matrix init")
    id_x = np.zeros((nodeCount, nodeCount), dtype=np.int32)
    id_y = np.zeros((nodeCount, nodeCount), dtype=np.int32)
    for i in range(nodeCount):
        matrix_x[i, :] *= int(points[i].x)
        matrix_y[i, :] *= int(points[i].y)
    print("multiplied done")
    matrix_x = matrix_x-np.transpose(matrix_x)
    matrix_y = matrix_y-np.transpose(matrix_y)
    distances = np.abs(matrix_x)+np.abs(matrix_y)
    sorted_distance = np.argsort(distances, axis=1)
    print(sorted_distance.shape)
    return sorted_distance, distances


def closest_greedy(nodeCount: int, points: List[Point2D]):
    sd, d = build_matrice_distance_np(nodeCount, points)
    g = nx.DiGraph()
    sol = [0]
    length_circuit = 0.
    index_in_sol = set([0])
    s = sd[:, 1:]
    # print(matrix.shape)
    # print(s.shape)
    # print(s)
    cur_point = 0
    nb_point = 1
    while nb_point<nodeCount:
        n = next(p for p in s[cur_point, :] if p not in index_in_sol)
        length_circuit += length(points[cur_point], points[n])
        index_in_sol.add(n)
        sol += [n]
        cur_point = n
        nb_point += 1
    length_circuit += length(points[cur_point], points[0])
    return sol, length_circuit, 0


def testing():
    file_location = "./data/tsp_574_1"
    #file_location = "./data/vrp_200_16_1"
    print(file_location)
    with open(file_location, 'r') as input_data_file:
        input_data = input_data_file.read()
        nodeCount, points = parse(input_data)
        sol, length_circuit, opt = closest_greedy(nodeCount, points)
        fig, ax = plt.subplots(1)
        pp, = ax.plot([points[jj].x for jj in sol], [points[jj].y for jj in sol])
        plt.show()


if __name__ == "__main__":
    testing()