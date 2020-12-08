from skdecide.builders.discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction, \
    build_aggreg_function_and_params_objective
from skdecide.builders.discrete_optimization.generic_tools.lns_mip import PostProcessSolution
from skdecide.builders.discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from skdecide.builders.discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import MS_RCPSPModel, MS_RCPSPModel_Variant, MS_RCPSPSolution, \
    MS_RCPSPSolution_Variant
import networkx as nx
from copy import deepcopy


class PostProMSRCPSP(PostProcessSolution):
    def __init__(self, problem: MS_RCPSPModel, params_objective_function: ParamsObjectiveFunction):
        self.problem = problem
        self.params_objective_function = params_objective_function
        self.aggreg_from_sol, self.aggreg_dict, self.params_objective_function = \
            build_aggreg_function_and_params_objective(problem=self.problem,
                                                       params_objective_function=
                                                       self.params_objective_function)
        self.graph = self.problem.compute_graph()

        self.successors = {n: nx.algorithms.descendants(self.graph.graph_nx, n)
                           for n in self.graph.graph_nx.nodes()}
        self.predecessors = {n: nx.algorithms.descendants(self.graph.graph_nx, n)
                             for n in self.graph.graph_nx.nodes()}
        self.immediate_predecessors = {n: self.graph.get_predecessors(n)
                                       for n in self.graph.nodes_name}

    def build_other_solution(self, result_storage: ResultStorage) -> ResultStorage:
        new_solution = sgs_variant(solution=result_storage.get_best_solution(),
                                   problem=self.problem,
                                   predecessors_dict=self.immediate_predecessors)
        fit = self.aggreg_from_sol(new_solution)
        result_storage.add_solution(new_solution, fit)
        import random
        for s in random.sample(result_storage.list_solution_fits,
                               min(len(result_storage.list_solution_fits), 50)):
            new_solution = sgs_variant(solution=s[0],
                                       problem=self.problem,
                                       predecessors_dict=self.immediate_predecessors)
            fit = self.aggreg_from_sol(new_solution)
            result_storage.add_solution(new_solution, fit)

        return result_storage


def sgs_variant(solution: MS_RCPSPSolution,
                problem: MS_RCPSPModel,
                predecessors_dict):
    task_that_be_shifted_left = []
    new_proposed_schedule = {}
    # 1, 2
    resource_avail_in_time = {}
    new_horizon = problem.horizon
    for res in problem.resources_set:
        resource_avail_in_time[res] = problem.resources_availability[res][:new_horizon + 1]
    worker_avail_in_time = {}
    for i in problem.employees:
        worker_avail_in_time[i] = list(problem.employees[i].calendar_employee)
    sorted_tasks = sorted(solution.schedule.keys(),
                          key=lambda x: (solution.schedule[x]["start_time"], x))
    for task in sorted_tasks:
        employee_used = [emp for emp in solution.employee_usage.get(task, {})
                         if len(solution.employee_usage[task][emp]) > 0]
        times_predecessors = [new_proposed_schedule[t]["end_time"]
                              if t in new_proposed_schedule
                              else solution.schedule[t]["end_time"]
                              for t in predecessors_dict[task]]
        if len(times_predecessors) > 0:
            min_time = max(times_predecessors)
        else:
            min_time = 0
        new_starting_time = solution.schedule[task]["start_time"]
        for t in range(min_time, solution.schedule[task]["start_time"]+1):
            if all(all(worker_avail_in_time[emp][time]
                       for time in range(t,
                                         t+ solution.schedule[task]["end_time"]-solution.schedule[task]["start_time"]))
                   for emp in employee_used):
                new_starting_time = t
                break
        duration = solution.schedule[task]["end_time"]-solution.schedule[task]["start_time"]
        new_proposed_schedule[task] = {"start_time": new_starting_time,
                                       "end_time": new_starting_time+duration}
        for t in range(new_starting_time, new_starting_time+duration):
            for emp in employee_used:
                worker_avail_in_time[emp][t] = False
    new_solution = MS_RCPSPSolution(problem=problem,
                                    modes=solution.modes,
                                    schedule=new_proposed_schedule,
                                    employee_usage=solution.employee_usage)
    #print("New : ", problem.evaluate(new_solution), problem.satisfy(new_solution))
    #print("Old : ", problem.evaluate(solution), problem.satisfy(solution))
    return new_solution







