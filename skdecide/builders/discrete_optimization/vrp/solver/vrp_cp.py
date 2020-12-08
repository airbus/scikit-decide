from skdecide.builders.discrete_optimization.vrp.solver_toolbox import Customer, compute_length_matrix
from skdecide.builders.discrete_optimization.vrp.vrp_toolbox import parse_input
from minizinc import Instance, Model, Solver, Status, Result
import os
this_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(this_path)

#
# int: nb_customers_without_depot;
# int: nb_vehicle;
# int: nb_customers_virtual = nb_customers_without_depot+nb_vehicle+1;
# set of int: customer_range=1..nb_customers_virtual;
# array[1..nb_vehicle] of customer_range: virtual_nodes = [nb_customers_virtual+i | i in 1..nb_vehicle+1];
# array[customer_range, customer_range] of int: distance;
# array[customer_range] of var customer_range: circuit_vrp;
# array[1..nb_vehicle] of var customer_range: index_end_subcircuit;
from datetime import timedelta
import random
def cp_solving(vehicle_count, vehicle_capacity, customer_count, customers, fraction=0.1):
    model_vrp = Model("./vrp.mzn")
    # Find the MiniZinc solver configuration for Gecode
    solver = Solver.lookup("chuffed")
    instance = Instance(solver, model_vrp)
    instance["nb_customers_without_depot"] = customer_count-1
    nb_customers_virtual = customer_count-1+vehicle_count+1
    range_customer_virtual = range(1, nb_customers_virtual+1)
    instance["nb_vehicle"] = vehicle_count
    closest, matrix = compute_length_matrix(customers)
    demands = [customers[i].demand for i in range(1, customer_count)]+[0]*(vehicle_count+1)
    instance["demand"] = demands
    instance["capacity_vehicle"] = vehicle_capacity
    recomputed_dist = []
    for i in range(1, customer_count):
        recomputed_dist += [[int(matrix[i, j]) for j in range(1, customer_count)]+[int(matrix[i, 0])]*(vehicle_count+1)]
    for v in range(vehicle_count+1):
        recomputed_dist += [[int(matrix[0, j]) for j in range(1, customer_count)]+[0]*(vehicle_count+1)]
    instance["distance"] = recomputed_dist
    result = instance.solve(timeout=timedelta(seconds=100))
    opt: Status = result.status
    print(result.__dict__)
    objective = result["objective"]
    print("Objective : ", objective)
    print(result["circuit_vrp"])
    iteration = 0
    dict_result = {i+1: result["circuit_vrp"][i] for i in range(nb_customers_virtual)}
    current_objective = objective
    while iteration < 1000:
        with instance.branch() as child:
            subpart_tsp = set(random.sample(range_customer_virtual, int(fraction*len(range_customer_virtual))))
            for i in range_customer_virtual:
                if i not in subpart_tsp:
                    # print("constraint color_graph["+str(i)+"] == "+ str(dict_color[i])+";\n")
                    child.add_string("constraint circuit_vrp["+str(i)+"] == "+ str(dict_result[i])+";\n")
            child.add_string(f"solve minimize(objective);\n")
            res = child.solve(timeout=timedelta(seconds=50))
            print(res.status)
            if res.solution is not None and res["objective"] <= current_objective:
                iteration += 1
                incumbent = res.solution
                current_objective = res["objective"]
                dict_result = {i+1: res["circuit_vrp"][i] for i in range(nb_customers_virtual)}
                #json.dump(dict_color, open('debug_cp.json', "w"))
                print(iteration , " : , ", res["objective"])
                print('IMPROVED : ', iteration)
                print("dict result : ", dict_result)
            else:
                try:
                    print("Not improved ")
                    print(iteration , " :  ", res["objective"])
                except:
                    print(iteration, " failed ")
                # print({i: res["color_graph"][i-1] for i in range_node})
                iteration += 1

def testing():
    #278.73
    #607.65
    #524.61
    #819.56
    #1322.12
    #1858.44
    # Q7wCC, ./data/vrp_16_3_1, solver.py, Vehicle Routing Problem 1
    # 5c314, ./data/vrp_26_8_1, solver.py, Vehicle Routing Problem 2
    # nUAU4, ./data/vrp_51_5_1, solver.py, Vehicle Routing Problem 3
    # QQs4z, ./data/vrp_101_10_1, solver.py, Vehicle Routing Problem 4
    # EV1zA, ./data/vrp_200_16_1, solver.py, Vehicle Routing Problem 5
    # 5U5So, ./data/vrp_421_41_1, solver.py, Vehicle Routing Problem 6
    file_location = "./data/vrp_101_10_1"
    #file_location = "./data/vrp_200_16_1"
    print(file_location)
    with open(file_location, 'r') as input_data_file:
        input_data = input_data_file.read()
        vehicle_count, vehicle_capacity, customer_count, customers = parse_input(input_data)
        cp_solving(vehicle_count, vehicle_capacity, customer_count, customers, fraction=0.5)

if __name__ == '__main__':
    testing()