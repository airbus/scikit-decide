#from numba import jit
from typing import List, Dict

from skdecide.builders.discrete_optimization.generic_tools.mip.pymip_tools import MyModelMilp
from skdecide.builders.discrete_optimization.vrp.vrp_toolbox import length
import networkx as nx
import os
import random
this_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(this_path)
import matplotlib.pyplot as plt
from mip import Model, CBC, MINIMIZE, xsum, BINARY, Var, GRB
from skdecide.builders.discrete_optimization.generic_tools.do_solver import SolverDO, ResultStorage
from skdecide.builders.discrete_optimization.generic_tools.do_problem import build_aggreg_function_and_params_objective, \
    ParamsObjectiveFunction
from skdecide.builders.discrete_optimization.vrp.vrp_model import VrpProblem, VrpSolution, compute_length
from copy import deepcopy
from skdecide.builders.discrete_optimization.vrp.solver.lp_vrp_iterative import build_graph_pruned_vrp, compute_start_end_flows_info


def init_model_lp(g,
                  edges,
                  edges_in_customers,
                  edges_out_customers,
                  edges_in_merged_graph,
                  edges_out_merged_graph,
                  edges_warm_set,
                  do_lns: False,
                  fraction: float,
                  start_indexes: List[int],
                  end_indexes: List[int],
                  vehicle_count: int,
                  vehicle_capacity: List[float],
                  include_backward=True,
                  include_triangle=False, solver_name=CBC):
    tsp_model = MyModelMilp("VRP-master", sense=MINIMIZE, solver_name=solver_name)
    x_var = {}  # decision variables on edges
    constraint_on_edge = {}
    edges_to_constraint = set()
    if do_lns:
        edges_to_constraint = set(random.sample(list(edges),
                                                int(fraction*len(edges))))
        for iedge in constraint_on_edge:
            tsp_model.remove(constraint_on_edge[iedge])
    iedge = 0
    start = []
    for e in edges:
        x_var[e] = tsp_model.add_var(var_type=BINARY,
                                     obj=g[e[0]][e[1]]["weight"],
                                     name="x_"+str(e))
        val = 0
        if e in edges_warm_set:
            start += [(x_var[e], 1)]
            val = 1
        else:
            start += [(x_var[e], 0)]
        if e in edges_to_constraint:
            constraint_on_edge[iedge] = tsp_model.add_constr(x_var[e] == val,
                                                             name=str((e, iedge)))
            iedge += 1
    tsp_model.start = start
    constraint_tour_2length = {}
    cnt_tour = 0
    if include_backward:
        for edge in edges:
            if (edge[1], edge[0]) in edges:
                if (edge[1], edge[0]) == edge:
                    continue
                if edge[0][1] == start_indexes[edge[0][0]] or edge[1][1] == start_indexes[edge[0][0]]:
                    continue
                if edge[0][1] == end_indexes[edge[0][0]] or edge[1][1] == end_indexes[edge[0][0]]:
                    continue
                constraint_tour_2length[cnt_tour] = tsp_model.add_constr(x_var[edge]+x_var[(edge[1],
                                                                                            edge[0])] <= 1,
                                                                         name="tour_"+str(edge))
                cnt_tour += 1
    if include_triangle:
        constraint_triangle = {}
        for node in g.nodes():
            neigh = set([n for n in nx.neighbors(g, node)])
            neigh_2 = {nn: neigh.intersection([n for n in nx.neighbors(g, nn)]) for nn in neigh}
            for node_neigh in neigh_2:
                if len(neigh_2[node_neigh]) >= 1:
                    for node_neigh_neigh in neigh_2[node_neigh]:
                        constraint_triangle[cnt_tour] = tsp_model.add_constr(
                            x_var[(node, node_neigh)]+x_var[(node_neigh, node_neigh_neigh)]
                            + x_var[(node_neigh_neigh, node)] <= 2
                        )
    #tsp_model.update()
    constraint_flow_in = {}
    constraint_flow_out = {}
    start_to_i, end_to_i = compute_start_end_flows_info(start_indexes, end_indexes)
    for s in start_to_i:
        for vehicle in start_to_i[s]["vehicle"]:
            constraint_flow_out["start_v_"+str(vehicle)] = \
                tsp_model.add_constr(xsum([x_var[e]
                                          for e in edges_out_customers[s]
                                          if e[0][0] == vehicle]) == 1,
                                     name="start_v_" + str(vehicle))
    for s in end_to_i:
        for vehicle in end_to_i[s]["vehicle"]:
            constraint_flow_in["end_v_"+str(vehicle)] = \
                tsp_model.add_constr(xsum([x_var[e]
                                          for e in edges_in_customers[s]
                                          if e[0][0] == vehicle]) == 1,
                                     name="end_v_" + str(vehicle))
    for customer in edges_in_customers:
        if customer in end_to_i or customer in start_to_i:
            # Already dealt by previous constraints
            continue
        else:
            constraint_flow_in[customer] = tsp_model.add_constr(xsum([x_var[e]
                                                                     for e in edges_in_customers[customer]]) == 1,
                                                                name="in_"+str(customer))
    # for customer in edges_out_customers:
    #     if customer in start_to_i or customer in end_to_i:
    #         # Already dealt by previous constraints
    #         continue
    #     else:
    #         constraint_flow_out[customer] = tsp_model.addConstr(quicksum([x_var[e]
    #                                                                       for e in edges_out_customers[customer]]) == 1,
    #                                                             name="outs_"+str(customer))
    c_flow = {}
    for n in edges_in_merged_graph:
        if start_indexes[n[0]] == end_indexes[n[0]] \
                or n[1] not in [start_indexes[n[0]], end_indexes[n[0]]]:
            c_flow[n] = tsp_model.add_constr(xsum([x_var[e] for e in edges_in_merged_graph[n]]
                                                     + [-x_var[e] for e in edges_out_merged_graph[n]]) == 0,
                                             name="flow_"+str(n))
    for v in range(vehicle_count):
        tsp_model.add_constr(xsum([g[e[0]][e[1]]["demand"]*x_var[e]
                                      for e in edges if e[0][0] == v])
                             <= vehicle_capacity[v],
                             name="capa_"+str(v))
    return tsp_model, x_var, constraint_flow_in, constraint_flow_out, constraint_on_edge


def update_graph(g, edges, 
                 edges_in_customers,
                 edges_out_customers,
                 edges_in_merged_graph,
                 edges_out_merged_graph, 
                 missing_edge,
                 customers):
    for edge in missing_edge:
        g.add_edge(edge[0], edge[1], weight=length(customers[edge[0][1]], 
                                                   customers[edge[1][1]]),
                   demand=customers[edge[1][1]].demand)
        g.add_edge(edge[1], edge[0], weight=length(customers[edge[0][1]], 
                                                       customers[edge[1][1]]),
                   demand=customers[edge[0][1]].demand)
        edges_in_merged_graph[edge[1]].add((edge[0], edge[1]))
        edges_out_merged_graph[edge[0]].add((edge[0], edge[1]))
        edges_in_customers[edge[1][1]].add((edge[0], edge[1]))
        edges_out_customers[edge[0][1]].add((edge[0], edge[1]))
        edges_in_merged_graph[edge[0]].add((edge[1], edge[0]))
        edges_out_merged_graph[edge[1]].add((edge[1], edge[0]))
        edges_in_customers[edge[0][1]].add((edge[1], edge[0]))
        edges_out_customers[edge[1][1]].add((edge[1], edge[0]))
        edges.add((edge[0], edge[1]))
        edges.add((edge[1], edge[0]))
    return g, edges, edges_in_customers, edges_out_customers, edges_in_merged_graph, edges_out_merged_graph


def build_warm_edges_and_update_graph(vrp_problem: VrpProblem,
                                      vrp_solution: VrpSolution,
                                      graph: nx.DiGraph,
                                      edges: set,
                                      edges_in_merged_graph,
                                      edges_out_merged_graph,
                                      edges_in_customers,
                                      edges_out_customers):
    vehicle_paths = deepcopy(vrp_solution.list_paths)
    edges_warm = []
    edges_warm_set = set()
    for i in range(len(vehicle_paths)):
        vehicle_paths[i] = [vrp_problem.start_indexes[i]]\
                            + vehicle_paths[i]+[vrp_problem.end_indexes[i]]
        edges_warm += [[((i, v1), (i, v2)) for v1, v2 in zip(vehicle_paths[i][:-1],
                                                             vehicle_paths[i][1:])]]
        edges_warm_set.update(set(edges_warm[i]))
        missing_edge = [e for e in set(edges_warm[i]) if e not in edges]
        for edge in missing_edge:
            graph.add_edge(edge[0], edge[1], weight=vrp_problem.evaluate_function_indexes(edge[0][1],
                                                                                          edge[1][1]),
                           demand=vrp_problem.customers[edge[1][1]].demand)
            graph.add_edge(edge[1], edge[0], weight=vrp_problem.evaluate_function_indexes(edge[1][1],
                                                                                          edge[0][1]),
                           demand=vrp_problem.customers[edge[0][1]].demand)
            edges_in_merged_graph[edge[1]].add((edge[0], edge[1]))
            edges_out_merged_graph[edge[0]].add((edge[0], edge[1]))
            edges_in_customers[edge[1][1]].add((edge[0], edge[1]))
            edges_out_customers[edge[0][1]].add((edge[0], edge[1]))
            edges_in_merged_graph[edge[0]].add((edge[1], edge[0]))
            edges_out_merged_graph[edge[1]].add((edge[1], edge[0]))
            edges_in_customers[edge[0][1]].add((edge[1], edge[0]))
            edges_out_customers[edge[1][1]].add((edge[1], edge[0]))
            edges.add((edge[0], edge[1]))
            edges.add((edge[1], edge[0]))
    return edges_warm, edges_warm_set


class VRPIterativeLP(SolverDO):
    def __init__(self, problem: VrpProblem, params_objective_function: ParamsObjectiveFunction):
        self.problem = problem
        self.model = None
        self.x_var = None
        self.constraint_on_edge = None
        self.aggreg_sol, self.aggreg_dict, self.params_objective_function = \
            build_aggreg_function_and_params_objective(problem=self.problem,
                                                       params_objective_function=params_objective_function)

    def init_model(self, **kwargs):
        g, g_empty, edges_in_customers, edges_out_customers, \
        edges_in_merged_graph, edges_out_merged_graph = \
            build_graph_pruned_vrp(self.problem)

        initial_solution = kwargs.get("initial_solution", None)
        if initial_solution is None:
            # solver = VrpORToolsSolver(self.problem)
            # solver.init_model()
            # solution, fit = solver.solve()
            solution = self.problem.get_dummy_solution()
        else:
            vehicle_tours_b = initial_solution
            solution = VrpSolution(problem=self.problem, list_start_index=self.problem.start_indexes,
                                   list_end_index=self.problem.end_indexes,
                                   list_paths=vehicle_tours_b,
                                   length=None,
                                   lengths=None,
                                   capacities=None)
        edges = set(g.edges())
        edges_warm, edges_warm_set = build_warm_edges_and_update_graph(vrp_problem=self.problem,
                                                                       vrp_solution=solution,
                                                                       graph=g,
                                                                       edges=edges,
                                                                       edges_in_merged_graph=edges_in_merged_graph,
                                                                       edges_out_merged_graph=edges_out_merged_graph,
                                                                       edges_in_customers=edges_in_customers,
                                                                       edges_out_customers=edges_out_customers)
        print(edges_warm, edges_warm_set)
        do_lns = kwargs.get("do_lns", False)
        fraction = kwargs.get("fraction_lns", 0.9)
        solver_name = kwargs.get("solver_name", CBC)
        tsp_model, x_var, constraint_flow_in, constraint_flow_out, constraint_on_edge = \
            init_model_lp(g=g,
                          edges=edges,
                          edges_in_customers=edges_in_customers,
                          edges_out_customers=edges_out_customers,
                          edges_in_merged_graph=edges_in_merged_graph,
                          edges_out_merged_graph=edges_out_merged_graph,
                          edges_warm_set=edges_warm_set,
                          start_indexes=self.problem.start_indexes,
                          end_indexes=self.problem.end_indexes,
                          do_lns=do_lns,
                          fraction=fraction,
                          vehicle_count=self.problem.vehicle_count,
                          vehicle_capacity=self.problem.vehicle_capacities,
                          solver_name=solver_name)
        self.model = tsp_model
        self.x_var = x_var
        self.constraint_on_edge = constraint_on_edge
        self.graph = g
        self.graph_infos = {"edges": edges,
                            "edges_in_customers": edges_in_customers,
                            "edges_out_customers": edges_out_customers,
                            "edges_in_merged_graph": edges_in_merged_graph,
                            "edges_out_merged_graph": edges_out_merged_graph,
                            "edges_warm_set": edges_warm_set}

    def solve(self, **kwargs):
        solver_name = kwargs.get("solver_name", CBC)
        do_lns = kwargs.get("do_lns", False)
        fraction = kwargs.get("fraction_lns", 0.9)
        nb_iteration_max = kwargs.get("nb_iteration_max", 20)
        if self.model is None:
            kwargs["solver_name"] = solver_name
            kwargs["do_lns"] = do_lns
            kwargs["fraction_lns"] = fraction
            self.init_model(**kwargs)
        print("optimizing...")
        limit_time_s = kwargs.get("limit_time_s", 10)
        self.model.optimize(max_seconds=limit_time_s)
        objective = self.model.objective_value
        # "C5t0ynWADsH8TEiH"
        # Query number of multiple objectives, and number of solutions
        finished = False
        solutions = []
        cost = []
        nb_components = []
        iteration = 0
        rebuilt_solution = []
        rebuilt_obj = []
        best_solution_rebuilt_index = 0
        best_solution_objective_rebuilt = float('inf')
        vehicle_count = self.problem.vehicle_count
        customers = self.problem.customers
        customer_count = self.problem.customer_count
        edges_in_customers = self.graph_infos["edges_in_customers"]
        edges_out_customers = self.graph_infos["edges_out_customers"]
        edges_in_merged_graph = self.graph_infos["edges_in_merged_graph"]
        edges_out_merged_graph = self.graph_infos["edges_out_merged_graph"]
        edges = self.graph_infos["edges"]
        edges_warm_set = self.graph_infos["edges_warm_set"]
        g = self.graph
        while not finished:
            solutions_ll = retreve_solutions(self.model,
                                             self.x_var,
                                             vehicle_count,
                                             g)
            solutions += [solutions_ll[0]["x_solution"]]
            cost += [objective]
            # print(solutions)
            x_solution, rebuilt_dict, \
            obj, components, components_global, component_all, component_global_all = \
                reevaluate_solutions(solutions_ll, vehicle_count, g,
                                     vrp_problem=self.problem)
            # for components_per_vehicle in component_all:
            #     update_model(self.problem,
            #                  self.model,
            #                  self.x_var,
            #                  components_per_vehicle,
            #                  edges_in_customers,
            #                  edges_out_customers)
            for comp in component_global_all:
                update_model_2(self.problem,
                               self.model,
                               self.x_var,
                               comp,
                               edges_in_customers,
                               edges_out_customers)

            nb_components += [len(components_global)]
            rebuilt_solution += [rebuilt_dict]
            rebuilt_obj += [obj]
            print('Objective rebuilt : ', rebuilt_obj[-1])
            if obj < best_solution_objective_rebuilt:
                best_solution_objective_rebuilt = obj
                best_solution_rebuilt_index = iteration
            iteration += 1
            if len(component_global_all[0]) > 1 or True:
                edges_to_add = set()
                for v in rebuilt_dict:
                    edges_to_add.update({(e0, e1) for e0, e1 in zip(rebuilt_dict[v][:-1], rebuilt_dict[v][1:])})
                    # print("len rebuilt : ", len(rebuilt_dict[v]))
                print("edges to add , ", edges_to_add)
                edges_missing = {e for e in edges_to_add if e not in edges}
                print("missing : ", edges_missing)

                if len(edges_missing) > 0:
                    g, edges, edges_in_customers, edges_out_customers, edges_in_merged_graph, edges_out_merged_graph = \
                        update_graph(g, edges, edges_in_customers,
                                     edges_out_customers,
                                     edges_in_merged_graph,
                                     edges_out_merged_graph,
                                     edges_missing,
                                     customers)
                    # self.model.reset()
                    self.model = None
                    tsp_model, x_var, constraint_flow_in, constraint_flow_out, constraint_on_edge = \
                        init_model_lp(g=g,
                                      edges=edges,
                                      edges_in_customers=edges_in_customers,
                                      edges_out_customers=edges_out_customers,
                                      edges_in_merged_graph=edges_in_merged_graph,
                                      edges_out_merged_graph=edges_out_merged_graph,
                                      edges_warm_set=edges_warm_set,
                                      start_indexes=self.problem.start_indexes,
                                      end_indexes=self.problem.end_indexes,
                                      do_lns=do_lns,
                                      fraction=fraction,
                                      vehicle_count=self.problem.vehicle_count,
                                      vehicle_capacity=self.problem.vehicle_capacities,
                                      solver_name=solver_name)
                    self.model = tsp_model
                    self.x_var = x_var
                    self.constraint_on_edge = constraint_on_edge
                    self.graph = g
                    self.graph_infos = {"edges": edges,
                                        "edges_in_customers": edges_in_customers,
                                        "edges_out_customers": edges_out_customers,
                                        "edges_in_merged_graph": edges_in_merged_graph,
                                        "edges_out_merged_graph": edges_out_merged_graph,
                                        "edges_warm_set": edges_warm_set}
                    edges_in_customers = self.graph_infos["edges_in_customers"]
                    edges_out_customers = self.graph_infos["edges_out_customers"]
                    edges_in_merged_graph = self.graph_infos["edges_in_merged_graph"]
                    edges_out_merged_graph = self.graph_infos["edges_out_merged_graph"]
                    edges = self.graph_infos["edges"]
                    edges_warm_set = self.graph_infos["edges_warm_set"]
                    for iedge in self.constraint_on_edge:
                        self.model.remove(self.constraint_on_edge[iedge])
                    self.model.update()
                    self.constraint_on_edge = {}
                edges_to_constraint = set(self.x_var.keys())
                if do_lns:
                    edges_to_constraint = set(random.sample(list(self.x_var.keys()),
                                                            int(fraction * len(self.x_var))))
                    for iedge in self.constraint_on_edge:
                        self.model.remove(self.constraint_on_edge[iedge])
                    self.model.update()
                    self.constraint_on_edge = {}
                    edges_to_constraint = set()
                    vehicle = set(random.sample(range(vehicle_count), min(4, vehicle_count)))
                    edges_to_constraint.update(set([e for e in edges if e[0][0] not in vehicle]))
                    # customers_to_constraint = set(random.sample(range(1, customer_count),
                    #                                             int(fraction * customer_count)))
                    # edges_to_constraint.update(set([edge for edge in edges
                    #                                 if (edge[0][1] in customers_to_constraint
                    #                                 or edge[1][1] in customers_to_constraint) ]))
                    print(len(edges_to_constraint), " edges constraint over ", len(edges))
                # print(rebuilt[0], rebuilt[-1])
                # print("len set rebuilt (debug) ", len(set(rebuilt_dict[v])))
                iedge = 0
                x_var = self.x_var
                start = []
                if all((e in edges) for e in edges_to_add):
                    # print("setting default value")
                    for e in x_var:
                        val = 0
                        if e in edges_to_add:
                            start += [(x_var[e], 1)]
                            val = 1
                        else:
                            start += [(x_var[e], 0)]
                        if e in edges_to_constraint:
                            if do_lns:
                                self.constraint_on_edge[iedge] = self.model.add_constr(x_var[e] == val,
                                                                                       name=str((e, iedge)))
                                iedge += 1
                    self.model.update()
                else:
                    pass
                # print([e for e in edges_to_add if e not in edges])
                self.model.start = start
                self.model.optimize(max_seconds=limit_time_s)
                objective = self.model.objective_value
            else:
                finished = True
            finished = finished or iteration >= nb_iteration_max

        plot = kwargs.get("plot", True)
        if plot:
            fig, ax = plt.subplots(2)
            for i in range(len(solutions)):
                ll = []
                for v in solutions[i]:
                    for e in solutions[i][v]:
                        ll.append(ax[0].plot([customers[e[0][1]].x, customers[e[1][1]].x],
                                             [customers[e[0][1]].y, customers[e[1][1]].y], color="b"))
                    ax[1].plot([customers[n[1]].x for n in rebuilt_solution[i][v]],
                               [customers[n[1]].y for n in rebuilt_solution[i][v]],
                               color="orange")
                ax[0].set_title("iter " + str(i) + " obj=" + str(int(cost[i])))
                ax[1].set_title("iter " + str(i) + " obj=" + str(int(rebuilt_obj[i])))
                # fig.savefig('./images/vrp_' + str(i) + ".png")
                plt.draw()
                plt.pause(0.01)
                ax[0].lines = []
                ax[1].lines = []
            plt.show()
        print("Best obj : ", best_solution_objective_rebuilt)
        print(rebuilt_obj[best_solution_rebuilt_index])
        solution = VrpSolution(problem=self.problem,
                               list_start_index=self.problem.start_indexes,
                               list_end_index=self.problem.end_indexes,
                               list_paths=[[x[1]
                                            for x in rebuilt_solution[best_solution_rebuilt_index][l][1:-1]]
                                           for l in sorted(rebuilt_solution[best_solution_rebuilt_index])],
                               length=None, lengths=None, capacities=None)
        fit = self.aggreg_sol(solution)
        return ResultStorage(list_solution_fits=[(solution, fit)],
                             mode_optim=self.params_objective_function.sense_function)


def build_the_cycles(x_solution,
                     component,
                     start_index,
                     end_index):
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
    while cur_edge[1] != end_node:
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
                        evaluate_function_indexes,
                        vrp_model: VrpProblem,
                        start_index,
                        end_index,
                        verbose=False):
    rebuilded_path = list(paths_component[node_to_component[start_index]])
    component_end = node_to_component[end_index]
    component_reconnected = {node_to_component[start_index]}
    current_component = sorted_connected_component[node_to_component[start_index]]
    path_set = set(rebuilded_path)
    total_length_path = len(rebuilded_path)
    print(rebuilded_path)
    while len(component_reconnected) < len(sorted_connected_component):
        if len(component_reconnected) == len(sorted_connected_component)-1 and \
            end_index != start_index and node_to_component[end_index] != node_to_component[start_index]:
            rebuilded_path = rebuilded_path+paths_component[component_end]
            component_reconnected.add(component_end)
        else:
            index_path = {rebuilded_path[i]: i for i in range(len(rebuilded_path))}
            index_path = {}
            for i in range(len(rebuilded_path)):
                if rebuilded_path[i] not in index_path:
                    index_path[rebuilded_path[i]] = []
                index_path[rebuilded_path[i]] += [i]
            edge_out_of_interest = {e
                                    for e in edges
                                    if e[0] in path_set
                                    and e[1] not in path_set}
            edge_in_of_interest = {e
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
                index_in = index_path[e[0]][0]
                # if index_in == total_length_path-1:
                #     continue
                index_in_1 = min(index_path[e[0]][0]+1, total_length_path-1)
                next_node_1 = rebuilded_path[index_in_1]
                component_e1 = node_to_component[e[1]]
                if component_e1 == component_end and len(component_reconnected) < len(sorted_connected_component)-1:
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
                graph.add_edge(e[0], e[1],
                               weight=evaluate_function_indexes(e[0][0], e[1][0]))
                graph.add_edge(e[1], e[0],
                               weight=evaluate_function_indexes(e[1][0], e[0][0]))
                min_out_edge = backup_min_out_edge
                min_in_edge = backup_min_in_edge
                min_index_in_path = backup_min_index_in_path
                min_component = backup_min_component
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
                print("New component : ", new_component)
            rebuilded_path = rebuilded_path[:(min_index_in_path+1)]+new_component+rebuilded_path[(min_index_in_path+1):]
            for e1, e2 in zip(new_component[:-1], new_component[1:]):
                if (e1, e2) not in graph.edges():
                    graph.add_edge(e1, e2, weight=evaluate_function_indexes(e1, e2))
            path_set = set(rebuilded_path)
            total_length_path = len(rebuilded_path)
            component_reconnected.add(min_component)
    print(rebuilded_path)
    lengths, obj, capacities = compute_length(start_index=start_index[1],
                                              end_index=end_index[1],
                                              solution=[x[1] for x in rebuilded_path[1:-1]],
                                              list_customers=vrp_model.customers,
                                              method=vrp_model.evaluate_function_indexes)

    return rebuilded_path, obj

from typing import Any
def retreve_solutions(model: Model,
                      x_var: Dict[Any, Var],
                      vehicle_count,
                      g):
    nSolutions = model.num_solutions
    # nObjectives = S.NumObj
    solutions = []
    # x = S.getVars()
    for s in range(nSolutions):
        # Set which solution we will query from now on
        g_empty = {v: nx.DiGraph() for v in range(vehicle_count)}
        g_merge = nx.DiGraph()
        x_solution = {v: set() for v in range(vehicle_count)}
        for e in x_var:
            value = x_var[e].xi(s)
            if value >= 0.5:
                vehicle = e[0][0]
                x_solution[vehicle].add(e)
                clients = e[0], e[1]
                if clients[0] not in g_empty[vehicle]: 
                    g_empty[vehicle].add_node(clients[0])
                if clients[1] not in g_empty[vehicle]: 
                    g_empty[vehicle].add_node(clients[1])
                if clients[0][1] not in g_merge: 
                    g_merge.add_node(clients[0][1])
                if clients[1][1] not in g_merge: 
                    g_merge.add_node(clients[1][1])
                g_empty[vehicle].add_edge(clients[0], clients[1], weight=g[e[0]][e[1]]["weight"])
                g_merge.add_edge(clients[0][1],
                                 clients[1][1],
                                 weight=g[e[0]][e[1]]["weight"])
        solutions += [{"g_merge": g_merge.copy(), "g_empty": g_empty, "x_solution": x_solution.copy()}]
    return solutions 


def reevaluate_solutions(solutions, vehicle_count, g, vrp_problem: VrpProblem):
    rebuilt_solution = []
    rebuilt_obj = []
    nb_components = []
    components = []
    components_global = []
    solutions_list = []
    print("Vehicle count : ", vehicle_count)
    for solution in solutions:
        g_empty = solution["g_empty"]
        g_merge = solution["g_merge"]
        x_solution = solution["x_solution"]
        connected_components = {v: [(set(e), len(e)) 
                                    for e in nx.weakly_connected_components(g_empty[v])]
                                for v in g_empty}
        print("Connected component : ", [len(connected_components[v]) for v in connected_components])
        print(x_solution)
        sorted_connected_component = {v: sorted(connected_components[v], key=lambda x: x[1], reverse=True)
                                      for v in connected_components}
        components += [sorted_connected_component]
        nb_components += [[len(sorted_connected_component[v]) for v in sorted_connected_component]]
        # print("NB COMPONENT", sum(nb_components[-1]))
        solutions_list += [x_solution.copy()]
        paths_component = {v: {} for v in range(vehicle_count)}
        indexes_component = {v: {} for v in range(vehicle_count)}
        node_to_component = {v: {} for v in range(vehicle_count)}
        nb_component = len(sorted_connected_component)
        rebuilt_dict = {}
        objective_dict = {}
        component_global = [(set(e), len(e)) 
                            for e in nx.weakly_connected_components(g_merge)] 
        components_global += [component_global]
        nb_component_global = len(component_global)
        print("Global : ", nb_component_global)
        for v in range(vehicle_count):
            graph_of_interest = nx.subgraph(g,
                                            [e[0] for e in x_solution[v]]
                                             + [e[1] for e in x_solution[v]]).copy()
            nb = len(sorted_connected_component[v])
            for i in range(nb):
                s = sorted_connected_component[v][i]
                paths_component[v][i], indexes_component[v][i] = build_the_cycles(x_solution=x_solution[v],
                                                                                  component=s[0],
                                                                                  start_index=(v,
                                                                                               vrp_problem.start_indexes[v]),
                                                                                  end_index=(v,
                                                                                             vrp_problem.end_indexes[v]))
                node_to_component[v].update({p: i for p in paths_component[v][i]})
            if True:
                rebuilt_dict[v], objective_dict[v] = rebuild_tsp_routine(sorted_connected_component=
                                                                         sorted_connected_component[v],
                                                                         paths_component=paths_component[v],
                                                                         node_to_component=node_to_component[v],
                                                                         start_index=(v, vrp_problem.start_indexes[v]),
                                                                         end_index=(v, vrp_problem.end_indexes[v]),
                                                                         indexes=indexes_component[v],
                                                                         graph=graph_of_interest,
                                                                         edges=set(graph_of_interest.edges()),
                                                                         evaluate_function_indexes=
                                                                         vrp_problem.evaluate_function_indexes,
                                                                         vrp_model=vrp_problem)
            else:
                rebuilt_dict[v], objective_dict[v] = None, float("inf")
        rebuilt_solution += [rebuilt_dict]
        rebuilt_obj += [sum(list(objective_dict.values()))]
    print("Rebuilt : ", rebuilt_solution, rebuilt_obj)
    index_best = min(range(len(rebuilt_obj)), key=lambda x: rebuilt_obj[x])
    print(index_best, "/", len(rebuilt_obj))
    print(rebuilt_obj)
    print("best : ", rebuilt_obj[index_best])
    return solutions_list[index_best], rebuilt_solution[index_best], rebuilt_obj[index_best], \
           components[index_best], components_global[index_best], components, components_global


def update_model(vrp_model: VrpProblem,
                 model: MyModelMilp,
                 x_var,
                 components_per_vehicle,
                 edges_in_customers,
                 edges_out_customers):
    for vehicle in components_per_vehicle:
        comps = components_per_vehicle[vehicle]
        print("Updating model : Nb component : ", len(comps))
        if len(comps) > 1:
            for si in comps:
                s = (set([x[1] for x in si[0]]),
                     si[1])
                print(vehicle, ":", s)
                edge_in_of_interest = [e for n in s[0]
                                       for e in edges_in_customers[n]
                                       if e[0][1] not in s[0]]
                edge_out_of_interest = [e for n in s[0]
                                        for e in edges_out_customers[n]
                                        if e[1][1] not in s[0]]
                print(edge_in_of_interest)
                print("Len of interest : ", len(edge_out_of_interest), len(edge_in_of_interest))
                model.add_constr(xsum([x_var[e] for e in edge_in_of_interest]) >= 1)
                model.add_constr(xsum([x_var[e] for e in edge_out_of_interest]) >= 1)
    model.update()


def update_model_2(vrp_model: VrpProblem,
                   model: MyModelMilp,
                   x_var,
                   components_global,
                   edges_in_customers,
                   edges_out_customers):
    len_component_global = len(components_global)
    if len_component_global > 1:
        print("Nb component : ", len_component_global)
        for s in components_global:
            # print(s)
            if True:
                edge_in_of_interest = [e for n in s[0] for e in edges_in_customers[n]
                                       if e[0][1] not in s[0] and e[1][1] in s[0]]
                edge_out_of_interest = [e for n in s[0] for e in edges_out_customers[n]
                                        if e[1][1] not in s[0] and e[0][1] in s[0]]
                # print(len(edge_out_of_interest), len(edge_in_of_interest))
                model.add_constr(xsum([x_var[e] for e in edge_in_of_interest]) >= 1)
                model.add_constr(xsum([x_var[e] for e in edge_out_of_interest]) >= 1)
    model.update()







    


