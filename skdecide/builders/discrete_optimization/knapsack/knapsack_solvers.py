from skdecide.builders.discrete_optimization.knapsack.solvers.greedy_solvers import GreedyBest, GreedyDummy
from skdecide.builders.discrete_optimization.knapsack.solvers.lp_solvers import KnapsackORTools, LPKnapsackCBC,\
    LPKnapsackGurobi, LPKnapsack, CBC
from skdecide.builders.discrete_optimization.knapsack.solvers.cp_solvers import CPKnapsackMZN, CPKnapsackMZN2
from skdecide.builders.discrete_optimization.knapsack.solvers.dyn_prog_knapsack import KnapsackDynProg
from skdecide.builders.discrete_optimization.knapsack.knapsack_model import KnapsackModel
from skdecide.builders.discrete_optimization.generic_tools.lp_tools import ParametersMilp
solvers = {"lp":[(KnapsackORTools, {}),
                 (LPKnapsackCBC, {}),
                 (LPKnapsackGurobi, {"parameter_gurobi": ParametersMilp.default()},
                 (LPKnapsack, {"solver_name": CBC, "limit_time_s": 10}))],
           "greedy": [(GreedyBest,  {})], 
           "cp": [(CPKnapsackMZN, {}),
                  (CPKnapsackMZN2, {})],
           "dyn_prog": [(KnapsackDynProg,  {'greedy_start': True, 
                                            'stop_after_n_item': True, 
                                            'max_items': 100, 
                                            'max_time_seconds': 100,
                                            'verbose': False})]}


def solve(method, 
          knapsack_model: KnapsackModel,
          **args):
    solver = method(knapsack_model)
    solver.init_model(**args)
    return solver.solve(**args)
