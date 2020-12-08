from abc import abstractmethod
from skdecide.builders.discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from skdecide.builders.discrete_optimization.generic_tools.do_solver import SolverDO
from enum import Enum
import mip


class MilpSolverName(Enum):
    CBC = 0
    GRB = 1


map_solver = {MilpSolverName.GRB: mip.GRB,
              MilpSolverName.CBC: mip.CBC}


class ParametersMilp:
    TimeLimit: int 
    PoolSolutions: int
    MIPGapAbs: float
    MIPGap: float
    retrieve_all_solution: bool
    pool_search_mode: int

    def __init__(self,
                 time_limit,
                 pool_solutions,
                 mip_gap_abs,
                 mip_gap,
                 retrieve_all_solution: bool,
                 n_solutions_max: int, pool_search_mode: int=0):
        self.TimeLimit = time_limit
        self.PoolSolutions = pool_solutions
        self.MIPGapAbs = mip_gap_abs
        self.MIPGap = mip_gap
        self.retrieve_all_solution = retrieve_all_solution
        self.n_solutions_max = n_solutions_max
        self.pool_search_mode = pool_search_mode

    @staticmethod
    def default():
        return ParametersMilp(time_limit=30,
                              pool_solutions=10000,
                              mip_gap_abs=0.0000001,
                              mip_gap=0.000001,
                              retrieve_all_solution=True,
                              n_solutions_max=10000)


class MilpSolver(SolverDO):
    @abstractmethod
    def init_model(self, **args):
        ...

    @abstractmethod
    def retrieve_solutions(self, parameters_milp: ParametersMilp) -> ResultStorage:
        ...

    @abstractmethod
    def solve(self, parameters_milp: ParametersMilp, **args) -> ResultStorage:
        ...
