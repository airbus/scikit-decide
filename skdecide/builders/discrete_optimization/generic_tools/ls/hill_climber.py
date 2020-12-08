import pickle
from typing import Optional, Union, List
from skdecide.builders.discrete_optimization.generic_tools.do_problem import Solution, Problem, \
    build_evaluate_function_aggregated, ObjectiveHandling, ModeOptim, ParamsObjectiveFunction
from skdecide.builders.discrete_optimization.generic_tools.ls.local_search import RestartHandler, \
                                                             ModeMutation, ResultLS
from skdecide.builders.discrete_optimization.generic_tools.do_mutation import Mutation
from skdecide.builders.discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage, ParetoFront


class HillClimber:
    def __init__(self, 
                 evaluator: Problem,
                 mutator: Mutation, 
                 restart_handler: RestartHandler,
                 mode_mutation: ModeMutation,
                 params_objective_function: ParamsObjectiveFunction=None,
                 store_solution=False,
                 nb_solutions=1000):
        self.evaluator = evaluator
        self.mutator = mutator
        self.restart_handler = restart_handler
        self.mode_mutation = mode_mutation
        self.params_objective_function = params_objective_function
        self.mode_optim = params_objective_function.sense_function
        self.aggreg_from_solution, self.aggreg_from_dict_values = \
            build_evaluate_function_aggregated(evaluator,
                                               params_objective_function=self.params_objective_function)
        self.store_solution = store_solution
        self.nb_solutions = nb_solutions

    def solve(self, 
              initial_variable: Solution,
              nb_iteration_max: int, 
              pickle_result=False,
              pickle_name="tsp")->ResultLS:
        objective = self.aggreg_from_dict_values(self.evaluator.evaluate(initial_variable))
        cur_variable = initial_variable.copy()
        if self.store_solution:
            store = ResultStorage(list_solution_fits=[(initial_variable, objective)],
                                  best_solution=initial_variable.copy(),
                                  limit_store=True,
                                  nb_best_store=1000)
        else:
            store = ResultStorage(list_solution_fits=[(initial_variable,
                                                       objective)],
                                  best_solution=initial_variable.copy(),
                                  limit_store=True,
                                  nb_best_store=1)
        cur_best_variable = initial_variable.copy()
        cur_objective = objective
        cur_best_objective = objective
        self.restart_handler.best_fitness = objective
        iteration = 0
        while iteration < nb_iteration_max:
            accept = False
            local_improvement = False
            global_improvement = False
            if self.mode_mutation == ModeMutation.MUTATE:
                nv, move = self.mutator.mutate(cur_variable)
                objective = self.aggreg_from_solution(nv)
            elif self.mode_mutation == ModeMutation.MUTATE_AND_EVALUATE:
                nv, move, objective = self.mutator.mutate_and_compute_obj(cur_variable)
                objective = self.aggreg_from_dict_values(objective)
            if self.mode_optim == ModeOptim.MINIMIZATION and objective < cur_objective:
                accept = True 
                local_improvement = True
                global_improvement = objective < cur_best_objective
            elif self.mode_optim == ModeOptim.MAXIMIZATION and objective > cur_objective:
                accept = True 
                local_improvement = True
                global_improvement = objective > cur_best_objective
            if accept:
                cur_objective = objective
                cur_variable = nv
            else:
                cur_variable = move.backtrack_local_move(nv)
            if self.store_solution:
                store.add_solution(nv, objective)
            if global_improvement: 
                print("iter ", iteration)
                print("new obj ", objective, " better than ", cur_best_objective)
                cur_best_objective = objective
                cur_best_variable = cur_variable.copy()
                if not self.store_solution:
                    store.add_solution(cur_variable, objective)
            # Update the temperature
            self.restart_handler.update(nv, objective, 
                                        global_improvement,
                                        local_improvement)
            # Update info in restart handler
            cur_variable, cur_objective = self.restart_handler.restart(cur_variable, cur_objective)
            # possibly restart somewhere
            iteration += 1
            if pickle_result and iteration % 20000 == 0:
                pickle.dump(cur_best_variable, open(pickle_name+".pk", "wb"))
        store.finalize()
        return store


class HillClimberPareto(HillClimber):
    def __init__(self,
                 evaluator: Problem,
                 mutator: Mutation,
                 restart_handler: RestartHandler,
                 mode_mutation: ModeMutation,
                 params_objective_function: ParamsObjectiveFunction=None,
                 store_solution=False,
                 nb_solutions=1000):
        super().__init__(evaluator=evaluator,
                         mutator=mutator,
                         restart_handler=restart_handler,
                         mode_mutation=mode_mutation,
                         params_objective_function=params_objective_function,
                         store_solution=store_solution,
                         nb_solutions=nb_solutions)

    def solve(self,
              initial_variable: Solution,
              nb_iteration_max: int,
              update_iteration_pareto=1000,
              pickle_result=False,
              pickle_name="tsp")->ResultLS:
        objective = self.aggreg_from_dict_values(self.evaluator.evaluate(initial_variable))
        pareto_front = ParetoFront(list_solution_fits=
                                   [(initial_variable, objective)],
                                   best_solution=initial_variable.copy(),
                                   limit_store=True,
                                   nb_best_store=1000)
        cur_variable = initial_variable.copy()
        cur_best_variable = initial_variable.copy()
        cur_objective = objective
        cur_best_objective = objective
        self.restart_handler.best_fitness = objective
        iteration = 0
        while iteration < nb_iteration_max:
            accept = False
            local_improvement = False
            global_improvement = False
            if iteration % update_iteration_pareto == 0:
                pareto_front.finalize()
            if self.mode_mutation == ModeMutation.MUTATE:
                nv, move = self.mutator.mutate(cur_variable)
                objective = self.aggreg_from_solution(nv)
            elif self.mode_mutation == ModeMutation.MUTATE_AND_EVALUATE:
                nv, move, objective = self.mutator.mutate_and_compute_obj(cur_variable)
                objective = self.aggreg_from_dict_values(objective)
            if self.mode_optim == ModeOptim.MINIMIZATION and objective < cur_objective:
                accept = True
                local_improvement = True
                global_improvement = objective < cur_best_objective
                pareto_front.add_solution(nv.copy(), objective)
            elif self.mode_optim == ModeOptim.MINIMIZATION and objective == cur_objective:
                accept = True
                local_improvement = True
                global_improvement = objective == cur_best_objective
                pareto_front.add_solution(nv.copy(), objective)
            elif self.mode_optim == ModeOptim.MAXIMIZATION and objective > cur_objective:
                accept = True
                local_improvement = True
                global_improvement = objective > cur_best_objective
                pareto_front.add_solution(nv.copy(), objective)
            elif self.mode_optim == ModeOptim.MAXIMIZATION and objective == cur_objective:
                accept = True
                local_improvement = True
                global_improvement = (objective == cur_best_objective)
                pareto_front.add_solution(nv.copy(), objective)
            if accept:
                print("Accept : ", objective)
                cur_objective = objective
                cur_variable = nv
            else:
                cur_variable = move.backtrack_local_move(nv)
            if global_improvement:
                print("iter ", iteration)
                print("new obj ", objective, " better than ", cur_best_objective)
                cur_best_objective = objective
                cur_best_variable = cur_variable.copy()
            # Update the temperature
            self.restart_handler.update(nv, objective,
                                        global_improvement,
                                        local_improvement)
            print("Len pareto : ", pareto_front.len_pareto_front())
            # Update info in restart handler
            cur_variable, cur_objective = self.restart_handler.restart(cur_variable, cur_objective)
            # possibly restart somewhere
            iteration += 1
            # if pickle_result and iteration % 20000 == 0:
            #    pickle.dump(cur_best_variable, open(pickle_name + ".pk", "wb"))
        pareto_front.finalize()
        return pareto_front


