from abc import abstractmethod
from skdecide.builders.discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage


class SolverDO:
    @abstractmethod
    def solve(self, **kwargs)->ResultStorage:
        ...
