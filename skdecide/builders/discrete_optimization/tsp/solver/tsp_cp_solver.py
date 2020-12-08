import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../../"))
from skdecide.builders.discrete_optimization.tsp.tsp_model import TSPModel, SolutionTSP
from skdecide.builders.discrete_optimization.tsp.common_tools_tsp import build_matrice_distance
from skdecide.builders.discrete_optimization.generic_tools.do_solver import SolverDO, ResultStorage
from skdecide.builders.discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction, \
    build_aggreg_function_and_params_objective
from minizinc import Instance, Model, Solver, Status, Result
from typing import List, Tuple
from datetime import timedelta
import os, random
this_path = os.path.dirname(os.path.abspath(__file__))


class TSP_CPModel:
    FLOAT_VERSION = 0
    INT_VERSION = 1


class TSP_CP_Solver(SolverDO):
    def __init__(self, tsp_model: TSPModel, 
                 model_type: TSP_CPModel,
                 params_objective_function: ParamsObjectiveFunction):
        self.tsp_model = tsp_model
        self.model_type = model_type
        self.start_index = self.tsp_model.start_index
        self.end_index = self.tsp_model.end_index
        self.instance = None
        self.key_decision_variable = ["x"]
        self.distance_matrix = build_matrice_distance(self.tsp_model.node_count, 
                                                      self.tsp_model.list_points,
                                                      method=self.tsp_model.evaluate_function_indexes)
        self.distance_matrix[self.end_index, self.start_index] = 0
        self.distance_list_2d = [[int(x) if model_type == TSP_CPModel.INT_VERSION else x
                                 for x in self.distance_matrix[i, :]]
                                 for i in range(self.distance_matrix.shape[0])]
        self.aggreg_sol, self.aggreg_dict, self.params_objective_function = \
            build_aggreg_function_and_params_objective(problem=self.tsp_model,
                                                       params_objective_function=params_objective_function)

    def init_model(self, **args):
        solver = args.get("solver", "chuffed")
        # Load n-Queens model from file
        if self.model_type == TSP_CPModel.FLOAT_VERSION:
            model = Model(os.path.join(this_path, "../minizinc/tsp_float.mzn"))
        if self.model_type == TSP_CPModel.INT_VERSION:
            model = Model(os.path.join(this_path, "../minizinc/tsp_int.mzn"))
        # Find the MiniZinc solver configuration for Gecode
        solver = Solver.lookup(solver)
        # Create an Instance of the n-Queens model for Gecode
        instance = Instance(solver, model)
        instance["n"] = self.tsp_model.node_count
        instance["distances"] = self.distance_list_2d
        instance["start"] = self.start_index+1
        instance["end"] = self.end_index+1
        self.instance = instance
                
    def solve(self, **args):
        max_time_seconds = args.get("max_time_seconds", 5)
        result = self.instance.solve(timeout=timedelta(seconds=max_time_seconds))
        print("Result = ", result)
        circuit = result["x"]
        start_index = self.start_index
        path = []
        cur_pos = self.start_index
        init = False
        while cur_pos!=self.end_index or not init:
            next_pos = circuit[cur_pos]-1
            path += [next_pos]
            cur_pos = next_pos
            init = True
        var_tsp = SolutionTSP(problem=self.tsp_model,
                              start_index=self.start_index,
                              end_index=self.end_index,
                              permutation=path[:-1],
                              length=None,
                              lengths=None)
        fit = self.aggreg_sol(var_tsp)
        return ResultStorage(list_solution_fits=[(var_tsp, fit)],
                             mode_optim=self.params_objective_function.sense_function)


