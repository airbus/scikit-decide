"""
Constraint programming common utilities and class that should be used by any solver using CP

"""
from abc import abstractmethod
from skdecide.builders.discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from skdecide.builders.discrete_optimization.generic_tools.do_solver import SolverDO
from enum import Enum


class CPSolverName(Enum):
    """
    Enum choice of underlying CP solver used by Minizinc typically
    """
    CHUFFED = 0
    GECODE = 1


map_cp_solver_name = {CPSolverName.CHUFFED: "chuffed",
                      CPSolverName.GECODE: "gecode"}


class ParametersCP:
    """
    Parameters that can be used by any cp - solver
    """
    TimeLimit: int
    TimeLimit_iter0: int
    PoolSolutions: int
    intermediate_solution: bool
    all_solutions: bool
    nr_solutions: int

    def __init__(self,
                 time_limit,
                 pool_solutions,
                 intermediate_solution: bool,
                 all_solutions: bool,
                 nr_solutions: int):
        """

        :param time_limit: in seconds, the time limit of solving the cp model
        :param pool_solutions: TODO remove it it's not used
        :param intermediate_solution: retrieve intermediate solutions
        :param all_solutions: returns all solutions found by the cp solver
        :param nr_solutions: max number of solution returned
        """
        self.TimeLimit = time_limit
        self.TimeLimit_iter0 = time_limit
        self.PoolSolutions = pool_solutions
        self.intermediate_solution = intermediate_solution
        self.all_solutions = all_solutions
        self.nr_solutions = nr_solutions

    @staticmethod
    def default():
        return ParametersCP(time_limit=30,
                            pool_solutions=10000,
                            intermediate_solution=True,
                            all_solutions=False,
                            nr_solutions=1000)


class CPSolver(SolverDO):
    """
    Additional function to be implemented by a CP Solver.
    """
    @abstractmethod
    def init_model(self, **args):
        """
        Instantiate a CP model instance
        """
        ...

    @abstractmethod
    def retrieve_solutions(self, result, parameters_cp: ParametersCP) -> ResultStorage:
        """
        Returns a storage solution coherent with the given parameters.
        :param result: Result storage returned by the cp solver
        :param parameters_cp: parameters of the CP solver.
        :return:
        """
        ...

    @abstractmethod
    def solve(self, parameters_cp: ParametersCP, **args) -> ResultStorage:
        ...
