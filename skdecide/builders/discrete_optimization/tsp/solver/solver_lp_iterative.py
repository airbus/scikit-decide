import os
from enum import Enum
#from numba import jit
import numpy as np
from skdecide.builders.discrete_optimization.generic_tools.do_solver import SolverDO, ResultStorage
from skdecide.builders.discrete_optimization.generic_tools.do_problem import build_evaluate_function_aggregated, \
    build_aggreg_function_and_params_objective, ParamsObjectiveFunction
from gurobi import Model, GRB, quicksum
from ortools.linear_solver import pywraplp
from skdecide.builders.discrete_optimization.tsp.tsp_model import TSPModel, SolutionTSP, TSPModel2D
from skdecide.builders.discrete_optimization.tsp.common_tools_tsp import build_matrice_distance, length, length_1, \
    build_matrice_distance_np, compute_length
folder_image = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../debug_image/image_lp_iterative/')
if not os.path.exists(folder_image):
    os.makedirs(folder_image)
import networkx as nx
import matplotlib.pyplot as plt


def build_graph_pruned(tsp_model: TSPModel2D):
    nodeCount = tsp_model.node_count
    points = tsp_model.list_points
    sd, d = build_matrice_distance_np(nodeCount, points)
    g = nx.DiGraph()
    g.add_nodes_from([i for i in range(nodeCount)])
    shape = sd.shape[0]
    edges_in = {i: set() for i in range(nodeCount)}
    edges_out = {i: set() for i in range(nodeCount)}
    def length_ij(i, j):
        return tsp_model.evaluate_function_indexes(i,j)
    for i in range(shape):
        nodes_to_add = sd[i, 1:50]
        for n in nodes_to_add:
            if n == i:
                continue
            g.add_edge(i, n, weight=length_ij(i,n))
            g.add_edge(n, i, weight=length_ij(n,i))
            edges_in[n].add((i, n))
            edges_out[i].add((i, n))
            edges_in[i].add((n, i))
            edges_out[n].add((n, i))
        nodes_to_add = range(i, min(i+5, nodeCount))
        for n in nodes_to_add:
            if n == i:
                continue
            g.add_edge(i, n, weight=length_ij(i,n))
            g.add_edge(n, i, weight=length_ij(n,i))
            edges_in[n].add((i, n))
            edges_out[i].add((i, n))
            edges_in[i].add((n, i))
            edges_out[n].add((n, i))
        nodes_to_add = [tsp_model.end_index]
        for n in nodes_to_add:
            if n == i:
                continue
            g.add_edge(i, n, weight=length_ij(i,n)) 
            g.add_edge(n, i, weight=length_ij(n,i))
            edges_in[n].add((i, n))
            edges_out[i].add((i, n))
            edges_in[i].add((n, i))
            edges_out[n].add((n, i))
    g_empty = nx.DiGraph()
    g_empty.add_nodes_from([i for i in range(nodeCount)])
    return g, g_empty, edges_in, edges_out


def build_graph_complete(tsp_model: TSPModel):
    nodeCount = tsp_model.node_count
    points = tsp_model.list_points
    mat = build_matrice_distance(nodeCount, points, tsp_model.evaluate_function_indexes)
    sd = np.argsort(mat, axis=1)
    g = nx.DiGraph()
    g.add_nodes_from([i for i in range(nodeCount)])
    shape = sd.shape[0]
    edges_in = {i: set() for i in range(nodeCount)}
    edges_out = {i: set() for i in range(nodeCount)}
    def length_ij(i, j):
        return mat[i, j]
    for i in range(shape):
        nodes_to_add = sd[i, 1:]
        for n in nodes_to_add:
            if n == i:
                continue
            g.add_edge(i, n, weight=length_ij(i, n))
            g.add_edge(n, i, weight=length_ij(n, i))
            edges_in[n].add((i, n))
            edges_out[i].add((i, n))
            edges_in[i].add((n, i))
            edges_out[n].add((n, i))
        nodes_to_add = [tsp_model.end_index]
        for n in nodes_to_add:
            if n == i:
                continue
            g.add_edge(i, n, weight=length_ij(i, n)) 
            g.add_edge(n, i, weight=length_ij(n, i))
            edges_in[n].add((i, n))
            edges_out[i].add((i, n))
            edges_in[i].add((n, i))
            edges_out[n].add((n, i))
    g_empty = nx.DiGraph()
    g_empty.add_nodes_from([i for i in range(nodeCount)])
    return g, g_empty, edges_in, edges_out


class MILPSolver(Enum):
    GUROBI = 0
    CBC = 1


class LP_TSP_Iterative(SolverDO):
    def __init__(self, tsp_model: TSPModel,
                 graph_builder=None,
                 params_objective_function: ParamsObjectiveFunction=None
                 ):
        self.tsp_model = tsp_model
        self.node_count = self.tsp_model.node_count
        self.list_points = self.tsp_model.list_points
        self.start_index = self.tsp_model.start_index
        self.end_index = self.tsp_model.end_index
        self.graph_builder = graph_builder
        self.g = None
        self.edges = None
        _, self.aggreg = build_evaluate_function_aggregated(tsp_model)
        self.aggreg_sol, self.aggreg, self.params_objective_function = \
            build_aggreg_function_and_params_objective(problem=self.tsp_model,
                                                       params_objective_function=params_objective_function)
    def init_model(self, method: MILPSolver):
        if method == MILPSolver.GUROBI:
            self.init_model_gurobi()
            self.method = method
        if method == MILPSolver.CBC:
            self.init_model_cbc()
            self.method = method

    def init_model_gurobi(self, **kwargs):
        g, g_empty, edges_in, edges_out = self.graph_builder(self.tsp_model)
        tsp_model = Model("TSP-master")
        edges = set(g.edges())
        self.edges = edges
        self.g = g
        x_var = {} # decision variables on edges
        dummy_sol = self.tsp_model.get_dummy_solution()
        path = [self.tsp_model.start_index]+dummy_sol.permutation+[self.tsp_model.end_index]
        edges_to_add = {(e0, e1)  for e0, e1 in zip(path[:-1], path[1:])}
        flow_in = {}
        flow_out = {}
        for e in edges:
            x_var[e] = tsp_model.addVar(vtype=GRB.BINARY, 
                                        obj=g[e[0]][e[1]]["weight"],
                                        name="x_"+str(e))
            if e[0] not in flow_out:
                flow_out[e[0]] = set()
            if e[1] not in flow_in:
                flow_in[e[1]] = set()
            flow_in[e[1]].add(e)
            flow_out[e[0]].add(e)
        if all((e in edges) for e in edges_to_add):
            for e in edges:
                if e in edges_to_add:
                    x_var[e].start = 1
                    x_var[e].varhintval = 1
                else:
                    x_var[e].start = 0
                    x_var[e].varhintval = 0
        constraint_tour_2length = {}
        cnt_tour = 0
        for edge in edges:
            if (edge[1], edge[0]) in edges:
                constraint_tour_2length[cnt_tour] = tsp_model.addConstr(x_var[edge]+x_var[(edge[1], edge[0])]<=1, name="Tour_"+str(cnt_tour))
                cnt_tour += 1
        tsp_model.update()
        # constraint_flow_in = {}
        # constraint_flow_out = {}
        constraint_flow = {}
        for n in flow_in:
            if n!=self.tsp_model.start_index and n!=self.tsp_model.end_index:
                constraint_flow[n] = tsp_model.addConstr(quicksum([x_var[i]
                                                                  for i in flow_in[n]]
                                                  +[-x_var[i] for i in flow_out[n]])==0, name="flow_"+str(n))
            if n != self.tsp_model.start_index:
                constraint_flow[(n, "sub")] = tsp_model.addConstr(quicksum([x_var[i]
                                                                            for i in flow_in[n]])==1, 
                                                                  name="flowin_"+str(n))
            if n==self.tsp_model.start_index:
                constraint_flow[(n, 0)] = tsp_model.addConstr(quicksum([x_var[i] for i in flow_out[n]])==1, name="flowoutsource_"+str(n))
                if n!=self.tsp_model.end_index:
                    constraint_flow[(n, 1)] = tsp_model.addConstr(quicksum([x_var[i] for i in flow_in[n]])==0, name="flowinsource_"+str(n))
            if n==self.tsp_model.end_index:
                constraint_flow[(n, 0)] = tsp_model.addConstr(quicksum([x_var[i] for i in flow_in[n]])==1, name="flowinsink_"+str(n))
                if n != self.tsp_model.start_index:
                    constraint_flow[(n, 1)] = tsp_model.addConstr(quicksum([x_var[i] for i in flow_out[n]])==0, name="flowoutsink_"+str(n))
        tsp_model.setParam("TimeLimit", 1000)
        tsp_model.modelSense = GRB.MINIMIZE
        tsp_model.setParam(GRB.Param.Threads, 8)
        tsp_model.setParam(GRB.Param.PoolSolutions, 10000)
        tsp_model.setParam(GRB.Param.Method, -1)
        tsp_model.setParam("MIPGapAbs", 0.001)
        tsp_model.setParam("MIPGap", 0.001)
        tsp_model.setParam("Heuristics", 0.1)

        self.model = tsp_model
        self.variables = {"x": x_var}
    
    def init_model_cbc(self, **kwargs):
        g, g_empty, edges_in, edges_out = self.graph_builder(self.tsp_model)
        tsp_model = pywraplp.Solver("TSP-master", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
        #S.EnableOutput()
        edges = set(g.edges())
        self.edges = edges
        self.g = g
        x_var = {} # decision variables on edges
        dummy_sol = self.tsp_model.get_dummy_solution()
        path = [self.tsp_model.start_index]+dummy_sol.permutation+[self.tsp_model.end_index]
        edges_to_add = {(e0, e1)  for e0, e1 in zip(path[:-1], path[1:])}
        flow_in = {}
        flow_out = {}
        for e in edges:
            x_var[e] = tsp_model.BoolVar("x_"+str(e))
            if e[0] not in flow_out:
                flow_out[e[0]] = set()
            if e[1] not in flow_in:
                flow_in[e[1]] = set()
            flow_in[e[1]].add(e)
            flow_out[e[0]].add(e)
        if all((e in edges) for e in edges_to_add):
            for e in edges:
                if e in edges_to_add:
                    tsp_model.SetHint([x_var[e]], [1])
                else:
                    tsp_model.SetHint([x_var[e]], [0])
        constraint_tour_2length = {}
        cnt_tour = 0
        for edge in edges:
            if (edge[1], edge[0]) in edges:
                constraint_tour_2length[cnt_tour] = tsp_model.Add(x_var[edge]+x_var[(edge[1], edge[0])]<=1)
                cnt_tour += 1
        # constraint_flow_in = {}
        # constraint_flow_out = {}
        constraint_flow = {}
        for n in flow_in:
            if n!=self.tsp_model.start_index and n!=self.tsp_model.end_index:
                constraint_flow[n] = tsp_model.Add(tsp_model.Sum([x_var[i]
                                                                  for i in flow_in[n]]
                                                  +[-x_var[i] for i in flow_out[n]])==0)
            if n != self.tsp_model.start_index:
                constraint_flow[(n, "sub")] = tsp_model.Add(tsp_model.Sum([x_var[i]
                                                            for i in flow_in[n]])==1)
            if n==self.tsp_model.start_index:
                constraint_flow[(n, 0)] = tsp_model.Add(tsp_model.Sum([x_var[i] for i in flow_out[n]])==1)
                if n!=self.tsp_model.end_index:
                    constraint_flow[(n, 1)] = tsp_model.Add(tsp_model.Sum([x_var[i] for i in flow_in[n]])==0)
            if n==self.tsp_model.end_index:
                constraint_flow[(n, 0)] = tsp_model.Add(tsp_model.Sum([x_var[i] for i in flow_in[n]])==1)
                if n != self.tsp_model.start_index:
                    constraint_flow[(n, 1)] = tsp_model.Add(tsp_model.Sum([x_var[i] for i in flow_out[n]])==0)
        value = tsp_model.Sum([x_var[i] * g[i[0]][i[1]]["weight"]
                            for i in x_var])
        tsp_model.Minimize(value)
        self.model = tsp_model
        self.variables = {"x": x_var}
        self.model.SetTimeLimit(60000)    

    def retrieve_results_cbc(self):
        g_empty = nx.DiGraph()
        g_empty.add_nodes_from([i for i in range(self.node_count)])
        x_solution = set()
        x_var = self.variables["x"]
        for e in x_var:
            value = x_var[e].solution_value()
            if value >= 0.5:
                x_solution.add(e)
                g_empty.add_edge(e[0], e[1], weight=1)
        return g_empty, x_solution
    
    def retrieve_results_gurobi(self):
        g_empty = nx.DiGraph()
        g_empty.add_nodes_from([i for i in range(self.node_count)])
        x_solution = set()
        x_var = self.variables["x"]
        for e in x_var:
            value = x_var[e].getAttr('X')
            if value >= 0.5:
                x_solution.add(e)
                g_empty.add_edge(e[0], e[1], weight=1)
        return g_empty, x_solution
    
    def solve(self, **kwargs):
        nb_iteration_max = kwargs.get("nb_iteration_max", 20)
        plot = kwargs.get("plot", True)
        tsp_model = self.model
        print("optimizing...")
        if self.method == MILPSolver.GUROBI:
            tsp_model.optimize()
            #"C5t0ynWADsH8TEiH"
            # Query number of multiple objectives, and number of solutions
            nSolutions = tsp_model.SolCount
            nObjectives = tsp_model.NumObj
            objective = tsp_model.getObjective().getValue()
            print('Problem has', nObjectives, 'objectives')
            print('Gurobi found', nSolutions, 'solutions')
            status = tsp_model.getAttr("Status")
        if self.method == MILPSolver.CBC:
            self.model.Solve()
            res = self.model.Solve()
            resdict = {0: 'OPTIMAL', 1: 'FEASIBLE', 2: 'INFEASIBLE', 3: 'UNBOUNDED',
                    4: 'ABNORMAL', 5: 'MODEL_INVALID', 6: 'NOT_SOLVED'}
            print('Result :', resdict[res])
            objective = self.model.Objective().Value()
        finished = False
        solutions = []
        cost = []
        nb_components = []
        iteration = 0
        rebuilt_solution = []
        rebuilt_obj = []
        best_solution_rebuilt_index = 0
        best_solution_rebuilt = float('inf')
        while not finished:
            if self.method == MILPSolver.GUROBI:
                g_empty, x_solution = self.retrieve_results_gurobi()
            if self.method == MILPSolver.CBC:
                g_empty, x_solution = self.retrieve_results_cbc()
            connected_components = [(set(e), len(e)) for e in nx.weakly_connected_components(g_empty)]
            print("Connected component : ", len(connected_components))
            sorted_connected_component = sorted(connected_components, key=lambda x: x[1], reverse=True)
            nb_components += [len(sorted_connected_component)]
            cost += [objective]
            solutions += [x_solution.copy()]
            paths_component = {}
            indexes_component = {}
            node_to_component = {}
            nb_component = len(sorted_connected_component)
            x_var = self.variables["x"]
            for i in range(nb_component):
                s = sorted_connected_component[i]
                paths_component[i], indexes_component[i] = build_the_cycles(x_solution, 
                                                                            s[0], 
                                                                            self.g, 
                                                                            start_index=self.start_index, 
                                                                            end_index=self.end_index)
                node_to_component.update({p: i for p in paths_component[i]})
                edge_in_of_interest = [e for e in self.edges if e[1] in s[0] and e[0] not in s[0]]
                edge_out_of_interest = [e for e in self.edges if e[0] in s[0] and e[1] not in s[0]]
                # if i <= len(sorted_connected_component)//2:
                if self.method == MILPSolver.GUROBI:
                    tsp_model.addConstr(quicksum([x_var[e] for e in edge_in_of_interest]) >= 1)
                    tsp_model.addConstr(quicksum([x_var[e] for e in edge_out_of_interest]) >= 1)
                if self.method == MILPSolver.CBC:
                    tsp_model.Add(tsp_model.Sum([x_var[e] for e in edge_in_of_interest]) >= 1)
                    tsp_model.Add(tsp_model.Sum([x_var[e] for e in edge_out_of_interest]) >= 1)
            print(len(node_to_component), self.node_count)
            print(len(x_solution))
            rebuilt, objective = rebuild_tsp_routine(sorted_connected_component, 
                                                     paths_component,
                                                     node_to_component, 
                                                     indexes_component,
                                                     self.g, 
                                                     self.edges, 
                                                     self.node_count,
                                                     self.list_points, 
                                                     self.tsp_model.evaluate_function_indexes, 
                                                     self.tsp_model,
                                                     self.start_index, 
                                                     self.end_index)
            objective = self.aggreg(objective)
            rebuilt_solution += [rebuilt]
            rebuilt_obj += [objective]
            if objective < best_solution_rebuilt:
                best_solution_rebuilt = objective
                best_solution_rebuilt_index = iteration
            if len(sorted_connected_component)>1:  
                edges_to_add = {(e0, e1)  for e0, e1 in zip(rebuilt[:-1], rebuilt[1:])}
                print("len rebuilt : ", len(rebuilt))
                #print(rebuilt[0], rebuilt[-1])
                print("len set rebuilt (debug) ", len(set(rebuilt)))
                if all((e in self.edges) for e in edges_to_add):
                    print("setting default value")
                    for e in x_var:
                        if e in edges_to_add:
                            if self.method == MILPSolver.GUROBI:
                                x_var[e].start = 1
                                x_var[e].varhintval = 1
                            elif self.method == MILPSolver.CBC:
                                tsp_model.SetHint([x_var[e]], [1])
                        else:
                            if self.method == MILPSolver.GUROBI:
                                x_var[e].start = 0
                                x_var[e].varhintval = 0
                            elif self.method == MILPSolver.CBC:
                                tsp_model.SetHint([x_var[e]], [1])
                else:
                    print([e for e in edges_to_add if e not in self.edges])
                if self.method == MILPSolver.GUROBI:
                    tsp_model.update()
                    tsp_model.optimize()
                if self.method == MILPSolver.CBC:
                    tsp_model.Solve()
                iteration += 1
            else:
                finished = True
            finished = finished or iteration>=nb_iteration_max
            if self.method == MILPSolver.GUROBI:
                objective = tsp_model.getObjective().getValue()
            elif self.method == MILPSolver.CBC:
                objective = self.model.Objective().Value()
            print("Objective : ", objective)
        if plot:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            for i in range(len(solutions)):
                ll = []
                for e in solutions[i]:
                    ll.append(ax[0].plot([self.list_points[e[0]].x, self.list_points[e[1]].x],
                                         [self.list_points[e[0]].y, self.list_points[e[1]].y], color="b"))
                ax[1].plot([self.list_points[n].x for n in rebuilt_solution[i]],
                           [self.list_points[n].y for n in rebuilt_solution[i]],
                           color="orange")
                ax[0].set_title("iter "+str(i)+" obj="+str(int(cost[i]))+" nbcomp="+str(nb_components[i]))
                ax[1].set_title("iter "+str(i)+" obj="+str(int(rebuilt_obj[i])))
                fig.savefig(os.path.join(folder_image, 'tsp_'+str(i)+".png"))
                plt.draw()
                plt.pause(1)
                ax[0].lines = []
                ax[1].lines = []
            plt.show()
        print("Best solution : ", best_solution_rebuilt)
        print(rebuilt_obj[best_solution_rebuilt_index])
        path = rebuilt_solution[best_solution_rebuilt_index]
        var_tsp = SolutionTSP(problem=self.tsp_model,
                              start_index=self.tsp_model.start_index,
                              end_index=self.tsp_model.end_index,
                              permutation=path[1:-1],
                              lengths=None,
                              length=None)
        fit = self.aggreg_sol(var_tsp)
        return ResultStorage(list_solution_fits=[(var_tsp, fit)],
                             mode_optim=self.params_objective_function.sense_function)


def build_the_cycles(x_solution, component, graph, start_index, end_index):
    edge_of_interest = {e for e in x_solution if e[1] in component and e[0] in component}
    innn = {e[1]: e for e in edge_of_interest}
    outt = {e[0]: e for e in edge_of_interest}
    if start_index in outt:
        some_node = start_index
    else:
        some_node = next(e[0] for e in edge_of_interest)
    end_node = some_node if end_index not in innn else end_index
    path = [some_node]
    cur_edge = outt[some_node]
    indexes = {some_node: 0}
    cur_index = 1
    while cur_edge[1]!=end_node:
        path += [cur_edge[1]]
        indexes[cur_edge[1]] = cur_index
        cur_index += 1
        cur_edge = outt[cur_edge[1]]
    if end_index in innn:
        path += [end_node]
        indexes[end_node] = cur_index
    return path, indexes


def rebuild_tsp_routine(sorted_connected_component, 
                        paths_component,
                        node_to_component, 
                        indexes,
                        graph, edges,
                        nodeCount, list_points,
                        evaluate_function_indexes,
                        tsp_model:TSPModel,
                        start_index=0, 
                        end_index=0,
                        verbose=False):
    print(len(node_to_component))
    rebuilded_path = list(paths_component[node_to_component[start_index]])
    component_end = node_to_component[end_index]
    component_reconnected = {node_to_component[start_index]}
    current_component = sorted_connected_component[node_to_component[start_index]]
    path_set = set(rebuilded_path)
    total_length_path = len(rebuilded_path)
    while len(component_reconnected)<len(sorted_connected_component):
        if len(component_reconnected) ==len(sorted_connected_component)-1 and \
            end_index!=start_index and node_to_component[end_index] != node_to_component[start_index]:
            rebuilded_path = rebuilded_path+paths_component[component_end]
            component_reconnected.add(component_end)
        else:
            index_path = {rebuilded_path[i]: i for i in range(len(rebuilded_path))}
            edge_out_of_interest = {e
                                    for e in edges 
                                    if e[0] in path_set 
                                    and e[1] not in path_set}
            edge_in_of_interest =  {e
                                    for e in edges 
                                    if e[0] not in path_set 
                                    and e[1] in path_set}
            min_out_edge = None
            min_in_edge = None
            min_index_in_path = None
            min_component = None
            min_dist = float('inf')
            backup_min_out_edge = None
            backup_min_in_edge = None
            backup_min_index_in_path = None
            backup_min_component = None
            backup_min_dist = float('inf')
            for e in edge_out_of_interest:
                index_in = index_path[e[0]]
                if index_in == total_length_path-1:
                    continue
                index_in_1 = index_path[e[0]]+1
                next_node_1 = rebuilded_path[index_in_1]
                component_e1 = node_to_component[e[1]]
                if component_e1 == component_end and len(component_reconnected)<len(sorted_connected_component)-1:
                    continue
                index_component_e1 = indexes[component_e1][e[1]]
                index_component_e1_plus1 = index_component_e1+1
                if index_component_e1_plus1 >= len(paths_component[component_e1]):
                    index_component_e1_plus1 = 0
                next_node_component_e1 = paths_component[component_e1][index_component_e1_plus1]
                if (next_node_component_e1, next_node_1) in edge_in_of_interest:
                    cost = graph[e[0]][e[1]]["weight"] + graph[next_node_component_e1][next_node_1]["weight"]-\
                            graph[e[0]][next_node_1]["weight"]
                    if cost < min_dist:
                        min_component = node_to_component[e[1]]
                        min_out_edge = e
                        min_in_edge = (next_node_component_e1, next_node_1)
                        min_index_in_path = index_in
                        min_dist = cost
                else:
                    cost = graph[e[0]][e[1]]["weight"]
                    if cost < backup_min_dist:
                        backup_min_component = node_to_component[e[1]]
                        backup_min_out_edge = e
                        backup_min_in_edge = (next_node_component_e1, next_node_1)
                        backup_min_index_in_path = index_in
                        backup_min_dist = cost
            if min_out_edge is None:
                print("Backup")
                e = backup_min_in_edge
                graph.add_edge(e[0], e[1], weight=evaluate_function_indexes(e[0], e[1]))
                graph.add_edge(e[1], e[0], weight=evaluate_function_indexes(e[1], e[0]))
                min_out_edge = backup_min_out_edge
                min_in_edge = backup_min_in_edge
                min_index_in_path = backup_min_index_in_path
                min_component = backup_min_component
                min_dist = backup_min_dist+graph[e[0]][e[1]]["weight"]-graph[min_in_edge[0]][e[1]]["weight"]
            len_this_component = len(paths_component[min_component])
            if verbose:
                print(list(range(0, -len_this_component, -1)))
                print("len this component : ", len_this_component)
                print("out edge :", min_out_edge)
                print("in edge :", min_in_edge)
            index_of_in_component = indexes[min_component][min_out_edge[1]]
            new_component = [paths_component[min_component][(index_of_in_component+i)%len_this_component] 
                            for i in range(0, -len_this_component, -1)]
            if verbose:
                print("path component ", paths_component[min_component])
                print("New compenent : ", new_component)

            rebuilded_path = rebuilded_path[:(min_index_in_path+1)]+new_component+rebuilded_path[(min_index_in_path+1):]
            for e1, e2 in zip(new_component[:-1], new_component[1:]):
                if (e1, e2) not in graph.edges():
                    graph.add_edge(e1, e2, weight=evaluate_function_indexes(e1, e2))
            path_set = set(rebuilded_path)
            total_length_path = len(rebuilded_path)
            component_reconnected.add(min_component)
    var = SolutionTSP(problem=tsp_model, start_index=start_index,
                      end_index=end_index, permutation=rebuilded_path[1:-1], lengths=None, length=None)
    fit = tsp_model.evaluate(var)
    print("ObjRebuilt=", fit)
    return rebuilded_path, fit




    


