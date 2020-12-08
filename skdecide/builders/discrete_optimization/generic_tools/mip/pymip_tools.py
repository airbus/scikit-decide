from typing import List, Tuple, Optional, Union
from mip import IncumbentUpdater, Var, Model, GRB, CBC, MINIMIZE, MAXIMIZE, BINARY, CONTINUOUS, INTEGER, xsum
import mip
import gc


def release_token():
    # Usefull if you are using a token licence of gurobi.. pymip is not well adapted to using those in loop (
    # (each new model instanciation will block a token as long your program runs..)
    # Running it will normally unblock the token before initalizing a new grb model.
    gc.collect()


class IncumbentStoreSolution(IncumbentUpdater):
    # Store intermediate solutions for further use.
    def __init__(self, model: Model):
        super().__init__(model=model)
        self._solution_store = []

    def nb_solutions(self):
        return len(self._solution_store)

    def get_solutions(self):
        return self._solution_store

    def update_incumbent(self, objective_value: float,
                         best_bound: float,
                         solution: List[Tuple[Var, float]]) -> List[Tuple[Var, float]]:
        dict_solution = {'obj': objective_value,
                         'best_bound': best_bound,
                         'solution': {var[0].name: var[1]
                                      for var in solution}}
        self._solution_store += [dict_solution]
        return solution


class MyModelMilp(Model):
    def __init__(
            self: "Model",
            name: str = "",
            sense: str = mip.MINIMIZE,
            solver_name: str = "",
            solver: Optional[mip.Solver] = None,
    ):
        super().__init__(name=name,
                         sense=sense,
                         solver_name=solver_name,
                         solver=solver)
        self.name = name
        self.sense = sense

    def remove(self: "MyModelMilp", objects: Union[mip.Var, mip.Constr, List[Union["mip.Var", "mip.Constr"]]]):
        super().remove(objects)
        self.update()

    def add_constr(self: "MyModelMilp", lin_expr: "mip.LinExpr", name: str = "") -> "mip.Constr":
        l = super().add_constr(lin_expr, name)
        self.update()
        return l

    def update(self):
        try:
            self.solver.update()
        except:
            pass
