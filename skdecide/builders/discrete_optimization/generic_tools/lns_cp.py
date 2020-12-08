from skdecide.builders.discrete_optimization.generic_tools.cp_tools import CPSolver, ParametersCP
from skdecide.builders.discrete_optimization.generic_tools.do_solver import SolverDO
from skdecide.builders.discrete_optimization.generic_tools.do_problem import Solution, Problem, build_evaluate_function_aggregated, \
    ParamsObjectiveFunction, ModeOptim, build_aggreg_function_and_params_objective
from skdecide.builders.discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from skdecide.builders.discrete_optimization.generic_tools.lns_mip import InitialSolution, PostProcessSolution, TrivialPostProcessSolution
from abc import abstractmethod
from typing import Any, Iterable, Optional
from datetime import timedelta
import time


class ConstraintHandler:
    @abstractmethod
    def adding_constraint_from_results_store(self,
                                             cp_solver: CPSolver,
                                             child_instance,
                                             result_storage: ResultStorage)->Iterable[Any]:
        ...

    @abstractmethod
    def remove_constraints_from_previous_iteration(self,
                                                   cp_solver: CPSolver,
                                                   child_instance,
                                                   previous_constraints: Iterable[Any]):
        ...



class LNS_CP(SolverDO):
    def __init__(self,
                 problem: Problem,
                 cp_solver: CPSolver,
                 initial_solution_provider: InitialSolution,
                 constraint_handler: ConstraintHandler,
                 post_process_solution: PostProcessSolution=None,
                 params_objective_function: ParamsObjectiveFunction=None):
        self.problem = problem
        self.cp_solver = cp_solver
        self.initial_solution_provider = initial_solution_provider
        self.constraint_handler = constraint_handler
        self.post_process_solution = post_process_solution
        if self.post_process_solution is None:
            self.post_process_solution = TrivialPostProcessSolution()
        self.params_objective_function = params_objective_function
        self.aggreg_from_sol, self.aggreg_dict, self.params_objective_function = \
            build_aggreg_function_and_params_objective(problem=self.problem,
                                                       params_objective_function=
                                                       self.params_objective_function)

    def solve_lns(self,
                  parameters_cp: ParametersCP,
                  nb_iteration_lns: int,
                  nb_iteration_no_improvement: Optional[int] = None,
                  max_time_seconds: Optional[int] = None,
                  skip_first_iteration: bool=False,
                  **args)->ResultStorage:
        sense = self.params_objective_function.sense_function
        if max_time_seconds is None:
            max_time_seconds = 3600*24  # One day
        if nb_iteration_no_improvement is None:
            nb_iteration_no_improvement = 2*nb_iteration_lns
        current_nb_iteration_no_improvement = 0
        deb_time = time.time()
        if not skip_first_iteration:
            store_lns = self.initial_solution_provider.get_starting_solution()
            init_solution, objective = store_lns.get_best_solution_fit()
            best_solution = init_solution.copy()

            satisfy = self.problem.satisfy(init_solution)
            print("Satisfy ", satisfy)
            best_objective = objective
        else:
            best_objective = float('inf') if sense==ModeOptim.MINIMIZATION else -float("inf")
            best_solution = None
            constraint_iterable = {"empty": []}
            store_lns = None
        for iteration in range(nb_iteration_lns):
            print('Starting iteration n°', iteration,
                  " current objective ", best_objective)
            with self.cp_solver.instance.branch() as child:
                if iteration == 0 and not skip_first_iteration or iteration >= 1:
                    constraint_iterable = self.constraint_handler \
                        .adding_constraint_from_results_store(cp_solver=self.cp_solver,
                                                              child_instance=child,
                                                              result_storage=store_lns)
                try:
                    if iteration == 0:
                        result = child.solve(timeout=timedelta(seconds=parameters_cp.TimeLimit_iter0),
                                             intermediate_solutions=parameters_cp.intermediate_solution)
                    else:
                        result = child.solve(timeout=timedelta(seconds=parameters_cp.TimeLimit),
                                             intermediate_solutions=parameters_cp.intermediate_solution)
                    result_store = self.cp_solver.retrieve_solutions(result, parameters_cp=parameters_cp)
                    print("iteration n°", iteration, "Solved !!!")
                    print(result.status)
                    if len(result_store.list_solution_fits) > 0:
                        print("Solved !!!")
                        bsol, fit = result_store.get_best_solution_fit()
                        print("Fitness = ", fit)
                        print("Post Process..")
                        result_store = self.post_process_solution.build_other_solution(result_store)
                        bsol, fit = result_store.get_best_solution_fit()
                        print("After postpro = ", fit)
                        if sense == ModeOptim.MAXIMIZATION and fit >= best_objective:
                            if fit > best_objective:
                                current_nb_iteration_no_improvement = 0
                            else:
                                current_nb_iteration_no_improvement += 1
                            best_solution = bsol
                            best_objective = fit
                        elif sense == ModeOptim.MAXIMIZATION:
                            current_nb_iteration_no_improvement += 1
                        elif sense == ModeOptim.MINIMIZATION and fit <= best_objective:
                            if fit < best_objective:
                                current_nb_iteration_no_improvement = 0
                            else:
                                current_nb_iteration_no_improvement += 1
                            best_solution = bsol
                            best_objective = fit
                        elif sense == ModeOptim.MINIMIZATION:
                            current_nb_iteration_no_improvement += 1
                        if skip_first_iteration and iteration == 0:
                            store_lns = result_store
                        for s, f in result_store.list_solution_fits:
                            store_lns.add_solution(solution=s, fitness=f)
                        print("Satisfy : ", self.problem.satisfy(best_solution))
                    else:
                        current_nb_iteration_no_improvement += 1
                except Exception as e:
                    current_nb_iteration_no_improvement += 1
                    print("Failed ! reason : ", e)
                if time.time() - deb_time > max_time_seconds:
                    print("Finish LNS with time limit reached")
                    break
                print(current_nb_iteration_no_improvement, "/", nb_iteration_no_improvement)
                if current_nb_iteration_no_improvement > nb_iteration_no_improvement:
                    print("Finish LNS with maximum no improvement iteration ")
                    break
                # Useless to remove the constraints with the "with", the constraints are only active inside the with.
                # print('Removing constraint:')
                # # self.constraint_handler.remove_constraints_from_previous_iteration(cp_solver=self.cp_solver,
                # #                                                                    child_instance=child,
                # #                                                                    previous_constraints=constraint_iterable)
                # print('Adding constraint:')
                # constraint_iterable = self.constraint_handler.adding_constraint_from_results_store(cp_solver=
                #                                                                                    self.cp_solver,
                #                                                                                    child_instance=child,
                #                                                                                    result_storage=
                #                                                                                    store_lns)
        return store_lns











