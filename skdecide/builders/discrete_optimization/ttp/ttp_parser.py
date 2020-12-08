import os, sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../"))
from skdecide.builders.discrete_optimization.ttp.ttp_model import TTPModel
folder_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/")
folder_data_gecco = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_gecco/")
list_files = [f for f in list(os.listdir(folder_data)) if f.endswith("ttp")]
list_files_gecco = [f for f in list(os.listdir(folder_data_gecco)) if f.endswith("txt")]

def get_data_available():
    return [os.path.join(folder_data, f) for f in list_files]+\
           [os.path.join(folder_data_gecco, f) for f in list_files_gecco]


def parse(file_path)->TTPModel:
    with open(file_path) as fp:
        line = fp.readline()
        cnt = 1
        problemName = None
        knapsackDataType = None
        numberOfNodes = None
        numberOfItems = None
        capacityOfKnapsack = None
        minSpeed = None
        maxSpeed = None
        rentingRatio = None
        edgeWeightType = None
        nodes = None
        items = None
        nodes_array = None
        items_array = None
        while line:
            #print("Line {}: {}".format(cnt, line.strip()))
            cnt += 1
            if line.startswith("PROBLEM NAME"):
                line = line[line.index(":")+1:]
                line = line.split()
                #line = line.replace(" ", "")
                problemName = line[0]
            elif line.startswith("KNAPSACK DATA TYPE"):
                line = line[line.index(":")+1:]
                line = line.split()
                #line = line.replace(" ", "")
                knapsackDataType = line[0]
            elif line.startswith("DIMENSION"):
                line = line[line.index(":")+1:]
                line = line.split()
                numberOfNodes = int(line[0])
            elif line.startswith("NUMBER OF ITEMS"):
                line = line[line.index(":")+1:]
                line = line.split()
                numberOfItems = int(line[0])
            elif line.startswith("CAPACITY OF KNAPSACK"):
                line = line[line.index(":")+1:]
                line = line.split()
                capacityOfKnapsack = float(line[0])
            elif line.startswith("MIN SPEED"):
                line = line[line.index(":")+1:]
                line = line.split()
                minSpeed = float(line[0])
            elif line.startswith("MAX SPEED"):
                line = line[line.index(":")+1:]
                line = line.split()
                maxSpeed = float(line[0])
            elif line.startswith("RENTING RATIO"):
                line = line[line.index(":")+1:]
                line = line.split()
                rentingRatio = float(line[0])
            elif line.startswith("EDGE_WEIGHT_TYPE"):
                line = line[line.index(":")+1:]
                line = line.split()
                edgeWeightType = line[0]
            elif line.startswith("NODE_COORD_SECTION"):
                nodes = []
                nodes_array = np.zeros((numberOfNodes, 3), dtype=np.int)
                for i in range(numberOfNodes):
                    line = fp.readline()
                    line = line.split()
                    significa = ["index", "x", 'y']
                    n = {}
                    for j in range(len(line)):
                        temp = int(line[j])
                        if j==0:
                            temp =  temp-1
                            temp = i
                        n[significa[j]] = temp
                        nodes_array[i, j] = temp
                    nodes += [n]
            elif line.startswith("ITEMS SECTION"):
                items = []
                items_array = np.zeros((numberOfItems, 4), dtype=np.int)
                for j in range(numberOfItems):
                    line = fp.readline()
                    line = line.split()
                    significa = ["index", "profit", "weight", "node_index"]
                    item = {}
                    for l in range(len(line)):
                        temp = int(line[l])
                        if l in {0, 3}:
                            temp = temp-1
                        if l == 0:
                            temp = j
                        item[significa[l]] = temp
                        items_array[j, l] = temp
                    items += [item]
            line = fp.readline()
        return TTPModel(problemName,
                        knapsackDataType,
                        numberOfNodes,
                        numberOfItems,
                        capacityOfKnapsack,
                        minSpeed,
                        maxSpeed,
                        rentingRatio,
                        edgeWeightType,
                        nodes, items, nodes_array, items_array)      

if __name__ == "__main__":
    one_file = os.path.join(folder_data, list_files[0])
    print(one_file)
    parse(one_file)