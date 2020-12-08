from gurobi import Model, GRB, quicksum
from mip import Model as MipModel
from mip import xsum, BINARY, INTEGER
import mip
import networkx as nx
from skdecide.builders.discrete_optimization.coloring.coloring_model import ColoringProblem, ColoringSolution
from skdecide.builders.discrete_optimization.coloring.solvers.greedy_coloring import GreedyColoring,\
    NXGreedyColoringMethod
from skdecide.builders.discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction,\
    build_aggreg_function_and_params_objective
from skdecide.builders.discrete_optimization.generic_tools.lp_tools import MilpSolver, ParametersMilp, MilpSolverName, map_solver
from skdecide.builders.discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage


class ColoringLP(MilpSolver):
    def __init__(self, coloring_problem: ColoringProblem,
                 params_objective_function: ParamsObjectiveFunction=None):
        self.coloring_problem = coloring_problem
        self.number_of_nodes = self.coloring_problem.number_of_nodes
        self.nodes_name = self.coloring_problem.graph.nodes_name
        self.index_nodes_name = {self.nodes_name[i]: i for i in range(self.number_of_nodes)}
        self.index_to_nodes_name = {i: self.nodes_name[i] for i in range(self.number_of_nodes)}
        self.graph = self.coloring_problem.graph
        self.model = None
        self.variable_decision = {}
        self.constraints_dict = {}
        self.description_variable_description = {}
        self.description_constraint = {}
        self.params_objective_function = params_objective_function
        self.aggreg_from_sol, self.aggreg_from_dict, self.params_objective_function = \
            build_aggreg_function_and_params_objective(problem=self.coloring_problem,
                                                       params_objective_function=params_objective_function)
        self.sense_optim = self.params_objective_function.sense_function
        self.start_solution = None

    def init_model(self, **kwargs):
        greedy_start = kwargs.get("greedy_start", True)
        verbose = kwargs.get("verbose", False)
        use_cliques = kwargs.get("use_cliques", False)
        if greedy_start:
            if verbose:
                print("Computing greedy solution")
            greedy_solver = GreedyColoring(self.coloring_problem,
                                           params_objective_function=self.params_objective_function)
            result_store = greedy_solver.solve(strategy=NXGreedyColoringMethod.best, verbose=verbose)
            self.start_solution = result_store.get_best_solution_fit()[0]
        else:
            if verbose:
                print("Get dummy solution")
            solution = self.coloring_problem.get_dummy_solution()
            self.start_solution = solution
        nb_colors = self.start_solution.nb_color
        color_model = Model("color")
        colors_var = {}
        range_node = range(self.number_of_nodes)
        range_color = range(nb_colors)
        for node in self.nodes_name:
            for color in range_color:
                colors_var[node, color] = color_model.addVar(vtype=GRB.BINARY,
                                                             obj=0,
                                                             name="x_" + str((node, color)))
        one_color_constraints = {}
        for n in range_node:
            one_color_constraints[n] = color_model.addConstr(quicksum([colors_var[n, c] for c in range_color]) == 1)
        color_model.update()
        cliques = []
        g = self.graph.to_networkx()
        if use_cliques:
            for c in nx.algorithms.clique.find_cliques(g):
                cliques += [c]
            cliques = sorted(cliques, key=lambda x: len(x), reverse=True)
        else:
            cliques = [[e[0], e[1]] for e in g.edges()]
        cliques_constraint = {}
        index_c = 0
        opt = color_model.addVar(vtype=GRB.INTEGER, lb=0, ub=nb_colors, obj=1)
        if use_cliques:
            for c in cliques[:100]:
                cliques_constraint[index_c] = color_model.addConstr(quicksum([(color_i + 1) * colors_var[node, color_i]
                                                                              for node in c
                                                                              for color_i in range_color])
                                                                    >= sum([i + 1 for i in range(len(c))]))
                cliques_constraint[(index_c, 1)] = color_model.addConstr(quicksum([colors_var[node, color_i]
                                                                                  for node in c
                                                                                  for color_i in range_color])
                                                                         <= opt)
                index_c += 1
        edges = g.edges()
        constraints_neighbors = {}
        for e in edges:
            for c in range_color:
                constraints_neighbors[(e[0], e[1], c)] = \
                    color_model.addConstr(colors_var[e[0], c] + colors_var[e[1], c] <= 1)
        for n in range_node:
            color_model.addConstr(quicksum([(color_i + 1) * colors_var[n, color_i] for color_i in range_color]) <= opt)
        color_model.update()
        color_model.modelSense = GRB.MINIMIZE
        color_model.setParam(GRB.Param.Threads, 8)
        color_model.setParam(GRB.Param.PoolSolutions, 10000)
        color_model.setParam(GRB.Param.Method, -1)
        color_model.setParam("MIPGapAbs", 0.001)
        color_model.setParam("MIPGap", 0.001)
        color_model.setParam("Heuristics", 0.01)
        self.model = color_model
        self.variable_decision = {"colors_var": colors_var}
        self.constraints_dict = {"one_color_constraints": one_color_constraints,
                                 "constraints_neighbors": constraints_neighbors}
        self.description_variable_description = {"colors_var": {"shape": (self.number_of_nodes, nb_colors),
                                                                "type": bool,
                                                                "descr": "for each node and each color,"
                                                                         " a binary indicator"}}
        self.description_constraint["one_color_constraints"] = {"descr": "one and only one color "
                                                                         "should be assignated to a node"}
        self.description_constraint["constraints_neighbors"] = {"descr": "no neighbors can have same color"}

    def retrieve_solutions(self, parameters_milp: ParametersMilp) -> ResultStorage:
        solution = [0] * self.number_of_nodes
        for key in self.variable_decision["colors_var"]:
            value = self.variable_decision["colors_var"][key].getAttr('X')
            if value >= 0.5:
                node = key[0]
                color = key[1]
                solution[self.index_nodes_name[node]] = color
        color_solution = ColoringSolution(self.coloring_problem, solution)
        fit = self.aggreg_from_sol(color_solution)
        return ResultStorage(list_solution_fits=[(color_solution, fit)],
                             best_solution=color_solution,
                             mode_optim=self.sense_optim)

    def solve(self, parameters_milp: ParametersMilp, **kwargs)->ResultStorage:
        if self.model is None:
            self.init_model(**kwargs)
        self.model.setParam("TimeLimit", parameters_milp.TimeLimit)
        self.model.optimize()
        n_solutions = self.model.SolCount
        n_objectives = self.model.NumObj
        objective = self.model.getObjective().getValue()
        print("Objective : ", objective)
        print('Problem has', n_objectives, 'objectives')
        print('Gurobi found', n_solutions, 'solutions')
        return self.retrieve_solutions(parameters_milp=parameters_milp)


class ColoringLP_MIP(ColoringLP):
    def __init__(self, coloring_problem: ColoringProblem,
                 params_objective_function: ParamsObjectiveFunction,
                 milp_solver_name: MilpSolverName=MilpSolverName.CBC):
        super().__init__(coloring_problem=coloring_problem,
                         params_objective_function=params_objective_function)
        self.milp_solver_name = milp_solver_name
        self.solver_name = map_solver[self.milp_solver_name]

    def init_model(self, **kwargs):
        greedy_start = kwargs.get("greedy_start", True)
        verbose = kwargs.get("verbose", False)
        use_cliques = kwargs.get("use_cliques", False)
        if greedy_start:
            if verbose:
                print("Computing greedy solution")
            greedy_solver = GreedyColoring(self.coloring_problem,
                                           params_objective_function=self.params_objective_function)
            result_store = greedy_solver.solve(strategy=NXGreedyColoringMethod.best, verbose=verbose)
            self.start_solution = result_store.get_best_solution_fit()[0]
        else:
            if verbose:
                print("Get dummy solution")
            solution = self.coloring_problem.get_dummy_solution()
            self.start_solution = solution
        nb_colors = self.start_solution.nb_color
        color_model = mip.Model("color",
                                sense=mip.MINIMIZE,
                                solver_name=self.solver_name)
        colors_var = {}
        range_node = range(self.number_of_nodes)
        range_color = range(nb_colors)
        for node in self.nodes_name:
            for color in range_color:
                colors_var[node, color] = color_model.add_var(var_type=BINARY,
                                                              obj=0,
                                                              name="x_" + str((node, color)))
        one_color_constraints = {}
        for n in range_node:
            one_color_constraints[n] = color_model.add_constr(xsum([colors_var[n, c] for c in range_color]) == 1)
        cliques = []
        g = self.graph.to_networkx()
        if use_cliques:
            for c in nx.algorithms.clique.find_cliques(g):
                cliques += [c]
            cliques = sorted(cliques, key=lambda x: len(x), reverse=True)
        else:
            cliques = [[e[0], e[1]] for e in g.edges()]
        cliques_constraint = {}
        index_c = 0
        opt = color_model.add_var(var_type=INTEGER, lb=0, ub=nb_colors, obj=1)
        if use_cliques:
            for c in cliques[:100]:
                cliques_constraint[index_c] = color_model.add_constr(xsum([(color_i + 1) * colors_var[node, color_i]
                                                                           for node in c
                                                                           for color_i in range_color])
                                                                     >= sum([i + 1 for i in range(len(c))]))
                cliques_constraint[(index_c, 1)] = color_model.add_constr(xsum([colors_var[node, color_i]
                                                                                for node in c
                                                                                for color_i in range_color])
                                                                          <= opt)
                index_c += 1
        edges = g.edges()
        constraints_neighbors = {}
        for e in edges:
            for c in range_color:
                constraints_neighbors[(e[0], e[1], c)] = \
                    color_model.add_constr(colors_var[e[0], c] + colors_var[e[1], c] <= 1)
        for n in range_node:
            color_model.add_constr(xsum([(color_i + 1) * colors_var[n, color_i] for color_i in range_color]) <= opt)
        self.model = color_model
        self.variable_decision = {"colors_var": colors_var}
        self.constraints_dict = {"one_color_constraints": one_color_constraints,
                                 "constraints_neighbors": constraints_neighbors}
        self.description_variable_description = {"colors_var": {"shape": (self.number_of_nodes, nb_colors),
                                                                "type": bool,
                                                                "descr": "for each node and each color,"
                                                                         " a binary indicator"}}
        self.description_constraint["one_color_constraints"] = {"descr": "one and only one color "
                                                                         "should be assignated to a node"}
        self.description_constraint["constraints_neighbors"] = {"descr": "no neighbors can have same color"}

    def retrieve_solutions(self, parameters_milp: ParametersMilp) -> ResultStorage:
        solution = [0] * self.number_of_nodes
        for key in self.variable_decision["colors_var"]:
            value = self.variable_decision["colors_var"][key].x
            if value >= 0.5:
                node = key[0]
                color = key[1]
                solution[self.index_nodes_name[node]] = color
        color_solution = ColoringSolution(self.coloring_problem, solution)
        fit = self.aggreg_from_sol(color_solution)
        return ResultStorage(list_solution_fits=[(color_solution, fit)],
                             best_solution=color_solution,
                             mode_optim=self.sense_optim)

    def solve(self, parameters_milp: ParametersMilp, **kwargs)->ResultStorage:
        if self.model is None:
            self.init_model(**kwargs)
        self.model.max_mip_gap = parameters_milp.MIPGap
        self.model.max_mip_gap_abs = parameters_milp.MIPGapAbs
        self.model.sol_pool_size = parameters_milp.PoolSolutions
        self.model.optimize(max_seconds=parameters_milp.TimeLimit,
                            max_solutions=parameters_milp.n_solutions_max)
        n_solutions = self.model.num_solutions
        objective = self.model.objective_value
        print("Objective : ", objective)
        print('Gurobi found', n_solutions, 'solutions')
        return self.retrieve_solutions(parameters_milp=parameters_milp)

