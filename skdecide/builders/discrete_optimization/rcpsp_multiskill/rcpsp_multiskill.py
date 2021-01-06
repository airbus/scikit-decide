# import os, sys
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../"))
from skdecide.builders.discrete_optimization.generic_tools.do_problem import Solution, Problem, EncodingRegister, TypeAttribute, \
    ObjectiveRegister, TypeObjective, ObjectiveHandling, TupleFitness, RobustProblem, MethodAggregating, ModeOptim
from skdecide.builders.discrete_optimization.generic_tools.graph_api import Graph
from typing import List, Union, NamedTuple, Tuple, Dict, Set, Optional
from abc import abstractmethod
import numpy as np
import scipy.stats as ss
from enum import Enum
from copy import deepcopy
import matplotlib.pyplot as plt
from collections import defaultdict


def tree():
    return defaultdict(tree)


class ScheduleGenerationScheme(Enum):
    SERIAL_SGS = 0
    PARALLEL_SGS = 1


class MS_RCPSPSolution(Solution):
    def __init__(self,
                 problem: Problem,
                 modes: Dict[int, int],
                 schedule: Dict[int, Dict[str, int]],  # (task: {"start_time": start, "end_time": }}
                 employee_usage: Dict[int, Dict[int, Set[str]]]):  # {task: {employee: Set(skills})}):
        self.problem: MS_RCPSPModel = problem
        self.modes = modes
        self.schedule = schedule
        self.employee_usage = employee_usage

    def copy(self):
        return MS_RCPSPSolution(problem=self.problem,
                                modes=deepcopy(self.modes),
                                schedule=deepcopy(self.schedule),
                                employee_usage=deepcopy(self.employee_usage))

    def change_problem(self, new_problem: Problem):
        self.problem = new_problem

def schedule_solution_to_variant(solution: MS_RCPSPSolution):
    s: MS_RCPSPSolution = solution
    priority_list_task = sorted(s.schedule, key=lambda x:
                                s.schedule[x]["start_time"])
    priority_list_task.remove(s.problem.source_task)
    priority_list_task.remove(s.problem.sink_task)
    workers = []
    for i in sorted(s.problem.tasks)[1:-1]:
        w = []
        if len(s.employee_usage.get(i, {})) > 0:
            w = [w for w in s.employee_usage.get(i)]
        w += [wi for wi in s.problem.employees if wi not in w]
        workers += [w]
    solution = MS_RCPSPSolution_Variant(problem=s.problem,
                                        priority_list_task=[p - 2 for p in priority_list_task],
                                        priority_worker_per_task=workers,
                                        modes_vector=[s.modes[i]
                                                      for i
                                                      in sorted(s.problem.tasks)[1:-1]])
    return solution


class MS_RCPSPSolution_Variant(MS_RCPSPSolution):
    def __init__(self, problem: Problem,
                 modes_vector: Optional[List[int]]=None,
                 modes_vector_from0: Optional[List[int]]=None,
                 priority_list_task: Optional[List[int]] = None,
                 priority_worker_per_task: Optional[List[List[int]]]=None,
                 modes: Dict[int, int]=None,
                 schedule: Dict[int, Dict[str, int]]=None,
                 employee_usage: Dict[int, Dict[int, Set[str]]]=None):  # {task: {employee: Set(skills})}):
        super().__init__(problem,
                         modes,
                         schedule,
                         employee_usage)
        self.priority_list_task = priority_list_task
        self.modes_vector = modes_vector
        self.modes_vector_from0 = modes_vector_from0
        if priority_worker_per_task is None:
            self.priority_worker_per_task = None
        elif any(isinstance(i, list) for i in priority_worker_per_task): # if arg is a nested list
            self.priority_worker_per_task = priority_worker_per_task
        else: # if arg is a single list
            self.priority_worker_per_task = self.problem.convert_fixed_priority_worker_per_task_from_permutation(priority_worker_per_task)

        if self.modes_vector is None and self.modes_vector_from0 is None:
            self.modes_vector = self.problem.fixed_modes
            self.modes_vector_from0 = [x-1 for x in self.problem.fixed_modes]
        if self.modes_vector is None and self.modes_vector_from0 is not None:
            self.modes_vector = [x+1 for x in self.modes_vector_from0]
        if self.modes_vector_from0 is None:
            self.modes_vector_from0 = [x-1 for x in self.modes_vector]
        if self.priority_list_task is None:
            self.priority_list_task = self.problem.fixed_permutation
        if self.priority_worker_per_task is None:
            self.priority_worker_per_task = self.problem.fixed_priority_worker_per_task
        self._schedule_to_recompute = True
        if self.schedule is None:
            self.do_recompute()

    def do_recompute(self):
        rcpsp_schedule, modes_extended, employee_usage, modes_dict = sgs_multi_skill(solution=self)
        self.schedule = rcpsp_schedule
        self.modes = modes_dict
        self.employee_usage = employee_usage
        self._schedule_to_recompute = False

    def copy(self):
        return MS_RCPSPSolution_Variant(problem=self.problem,
                                        priority_list_task=deepcopy(self.priority_list_task),
                                        modes_vector=deepcopy(self.modes_vector),
                                        priority_worker_per_task=deepcopy(self.priority_worker_per_task),
                                        modes=deepcopy(self.modes),
                                        schedule=deepcopy(self.schedule),
                                        employee_usage=deepcopy(self.employee_usage))


def sgs_multi_skill(solution: MS_RCPSPSolution_Variant):
    problem: MS_RCPSPModel = solution.problem
    perm = deepcopy(solution.priority_list_task)
    activity_end_times = {}

    unfeasible_non_renewable_resources = False
    unfeasible_in_horizon = False
    new_horizon = problem.horizon * problem.horizon_multiplier

    # 1, 2
    resource_avail_in_time = {}
    for res in problem.resources_set:
        resource_avail_in_time[res] = problem.resources_availability[res][:new_horizon + 1]
    worker_avail_in_time = {}
    for i in problem.employees:
        worker_avail_in_time[i] = list(problem.employees[i].calendar_employee)

    # 3
    minimum_starting_time = {}
    for act in list(problem.successors.keys()):
        minimum_starting_time[act] = 0
    perm_extended = [x + 2 for x in perm]
    perm_extended.insert(0, 1)
    perm_extended.append(problem.n_jobs_non_dummy + 2)
    # print('perm_extended: ', perm_extended)
    modes_extended = deepcopy(solution.modes_vector)
    modes_extended.insert(0, 1)
    modes_extended.append(1)
    modes_dict = {i+1: modes_extended[i] for i in range(len(modes_extended))}
    # print('pre-modes_extended: ', modes_extended)
    # fix modes in case specified mode not in mode details for the activites
    for i in range(len(modes_extended)):
        # print(list(self.problem.mode_details[i + 1].keys()))
        if modes_extended[i] not in list(problem.mode_details[i + 1].keys()):
            modes_extended[i] = 1
    # print('modes_extended: ', modes_extended)
    # print('start SGS')
    employee_usage = {}
    while len(perm_extended) > 0 and not unfeasible_non_renewable_resources and not unfeasible_in_horizon:
        # print('perm_extended: ', perm_extended)
        # get first activity in perm with precedences respected
        act_id = None
        for id in perm_extended:
            respected = True
            for pred in problem.successors.keys():
                if id in problem.successors[pred] and pred in perm_extended:
                    respected = False
                    break
            if respected:
                act_id = id
                break
        # print('next act_id respecting precedences :', act_id)
        # for act_id in perm_extended:  # 4
        # print(perm_extended, act_id)
        current_min_time = minimum_starting_time[act_id]  # 5
        # print('current_min_time_0: ', current_min_time)
        valid = False  # 6
        while not valid:  # 7
            # print('current_min_time: ', current_min_time)
            valid = True  # 8
            mode = modes_extended[act_id - 1]
            range_time = range(current_min_time,
                               current_min_time
                               + problem.mode_details[act_id][mode]['duration'])
            if current_min_time + problem.mode_details[act_id][mode]['duration'] >= problem.horizon:
                unfeasible_in_horizon = True
                break
            for t in range_time:  # 9
                # print(act_id, t)
                for res in resource_avail_in_time.keys():  # 10
                    if t < new_horizon:
                        if resource_avail_in_time[res][t] < \
                                problem.mode_details[act_id][modes_extended[act_id - 1]][res]:  # 11
                            valid = False  # 12
                            # print("ressource not found")
                            # print(res, act_id)
                            break
                    else:
                        unfeasible_non_renewable_resources = True
                        valid = False
                        # print('unfeasible ')
                        break
                if not valid:
                    break
            if valid:
                required_skills = {s: problem.mode_details[act_id][mode][s]
                                   for s in problem.mode_details[act_id][mode]
                                   if s in problem.skills_set and problem.mode_details[act_id][mode][s] > 0}
                worker_ids = None
                if len(required_skills) > 0:
                   worker_ids = [worker for worker in worker_avail_in_time
                                 if all(worker_avail_in_time[worker][t] for t in range_time)]
                   if problem.one_unit_per_task_max:
                       good = False
                       ws = []
                       for worker in worker_ids:
                           if any(s not in problem.employees[worker].dict_skill for s in required_skills) or \
                                   any(problem.employees[worker].dict_skill[s].skill_value<required_skills[s]
                                       for s in required_skills):
                               continue
                           else:
                               good = True
                               ws += [worker]
                               break
                       valid = good
                       if good:
                           worker_ids = ws
                   else:
                       if not all(sum([problem.employees[worker].dict_skill[s].skill_value
                                       for worker in worker_ids
                                       if s in problem.employees[worker].dict_skill]) >= required_skills[s]
                                  for s in required_skills):
                           valid = False
            if not valid:  # 13
                current_min_time += 1  # 14
            if current_min_time > new_horizon:
                unfeasible_in_horizon = True
                break
        if not unfeasible_non_renewable_resources and not unfeasible_in_horizon:
            # print('current_min_time: ', current_min_time)
            # print('in mode: ', modes_extended[act_id-1])
            # print('(mode details - duration: ', self.problem.mode_details[act_id][modes_extended[act_id-1]]['duration'])
            end_t = current_min_time + problem.mode_details[act_id][modes_extended[act_id - 1]]['duration'] - 1
            # print('end_t: ', end_t)
            for t in range(current_min_time,
                           current_min_time + problem.mode_details[act_id][modes_extended[act_id - 1]]['duration']):  # 15
                for res in resource_avail_in_time.keys():  # 16
                    resource_avail_in_time[res][t] -= problem.mode_details[act_id][modes_extended[act_id - 1]][res]  # 17
                    if res in problem.non_renewable_resources and t == end_t:
                        for tt in range(end_t + 1, new_horizon):
                            resource_avail_in_time[res][tt] -= \
                            problem.mode_details[act_id][modes_extended[act_id - 1]][res]
                            if resource_avail_in_time[res][tt] < 0:
                                unfeasible_non_renewable_resources = True
            if worker_ids is not None:
                priority_list_this_task = solution.priority_worker_per_task[act_id-2]
                worker_used = []
                current_skills = {s: 0. for s in required_skills}
                for w in priority_list_this_task:
                    if w in worker_ids:
                        worker_used += [w]
                        for s in problem.employees[w].dict_skill:
                            if s in current_skills:
                                current_skills[s] += problem.employees[w].dict_skill[s].skill_value
                                if act_id not in employee_usage:
                                    employee_usage[act_id] = {}
                                if w not in employee_usage[act_id]:
                                    employee_usage[act_id][w] = set()
                                employee_usage[act_id][w].add(s)
                        for t in range(current_min_time,
                                       current_min_time
                                       + problem.mode_details[act_id][modes_extended[act_id - 1]]['duration']):
                            worker_avail_in_time[w][t] = False
                    if all(current_skills[s] >= required_skills[s] for s in required_skills):
                        break
            activity_end_times[act_id] = current_min_time + \
                                         problem.mode_details[act_id][modes_extended[act_id - 1]]['duration']  # 18
            perm_extended.remove(act_id)
            # print('scheduled to complete at: ', activity_end_times[act_id])
            for s in problem.successors[act_id]:  # 19
                minimum_starting_time[s] = max(minimum_starting_time[s], activity_end_times[act_id])  # 20
        worker_ids = None
    # print('activity_end_times: ', activity_end_times)
    rcpsp_schedule = {}
    for act_id in activity_end_times:
        rcpsp_schedule[act_id] = {}
        rcpsp_schedule[act_id]['start_time'] = activity_end_times[act_id] - \
                                                    problem.mode_details[act_id][modes_extended[act_id - 1]][
                                                        'duration']
        rcpsp_schedule[act_id]['end_time'] = activity_end_times[act_id]
    if unfeasible_non_renewable_resources or unfeasible_in_horizon:
        rcpsp_schedule_feasible = False
        last_act_id = max(problem.successors.keys())
        if last_act_id not in rcpsp_schedule.keys():
            rcpsp_schedule[last_act_id] = {}
            rcpsp_schedule[last_act_id]['start_time'] = 99999999
            rcpsp_schedule[last_act_id]['end_time'] = 9999999
    else:
        rcpsp_schedule_feasible = True
    # print("unfeasible in horizon", unfeasible_in_horizon)
    # print(rcpsp_schedule[problem.sink_task]["end_time"])
    return rcpsp_schedule, modes_extended, employee_usage, modes_dict


class SkillDetail:
    skill_value: float
    efficiency_ratio: float
    experience: float

    def __init__(self, skill_value: float,
                 efficiency_ratio: float,
                 experience: float):
        self.skill_value = skill_value
        self.efficiency_ratio = efficiency_ratio
        self.experience = experience

    def copy(self):
        return SkillDetail(skill_value=self.skill_value,
                           efficiency_ratio=self.efficiency_ratio,
                           experience=self.experience)


class Employee:
    dict_skill: Dict[str, SkillDetail]
    calendar_employee: List[bool]

    def __init__(self, dict_skill: Dict[str, SkillDetail],
                 calendar_employee: List[bool], salary: float=0.):
        self.salary = salary
        self.dict_skill = dict_skill
        self.calendar_employee = calendar_employee

    def copy(self):
        return Employee(dict_skill={s: self.dict_skill[s].copy()
                                    for s in self.dict_skill},
                        calendar_employee=list(self.calendar_employee))


def intersect(i1, i2):
    if (i2[0] >= i1[1]
         or i1[0] >= i2[1]):
        return None
    else:
        s = max(i1[0], i2[0])
        e = min(i1[1], i2[1])
        return [s, e]

# Multiskill project scheduling
class MS_RCPSPModel(Problem):
    sgs: ScheduleGenerationScheme
    skills_set: Set[str]
    resources_set: Set[str]
    non_renewable_resources: Set[str]
    resources_availability: Dict[str, List[int]]
    employees: Dict[int,
                    Employee]
    employees_availability: List[int]

    n_jobs_non_dummy: int
    mode_details: Dict[int, Dict[int, Dict[str, int]]]
    successors: Dict[int, List[int]]

    def __init__(self,
                 skills_set: Set[str],
                 resources_set: Set[str],
                 non_renewable_resources: Set[str],
                 resources_availability:  Dict[str, List[int]],
                 employees: Dict[int,
                                 Employee],
                 employees_availability: List[int],
                 mode_details: Dict[int, Dict[int, Dict[str, int]]],
                 successors: Dict[int, List[int]],
                 horizon,
                 horizon_multiplier=1,
                 sink_task: Optional[int]=None,
                 source_task: Optional[int]=None,
                 one_unit_per_task_max: bool=False):
        self.skills_set = skills_set
        self.resources_set = resources_set
        self.resources_list = list(self.resources_set)
        self.non_renewable_resources = non_renewable_resources
        self.resources_availability = resources_availability
        self.employees = employees
        self.employees_availability = employees_availability
        self.mode_details = mode_details
        self.successors = successors
        self.n_jobs_non_dummy = len(self.mode_details.keys()) - 2
        self.horizon = horizon
        self.horizon_multiplier = horizon_multiplier
        self.sink_task = sink_task
        self.source_task = source_task
        if self.sink_task is None:
            self.sink_task = max(self.mode_details)
        if self.source_task is None:
            self.source_task = min(self.mode_details)
        self.nb_tasks = len(self.mode_details)
        self.tasks = list(sorted(self.mode_details.keys()))
        self.one_unit_per_task_max = one_unit_per_task_max

    def build_multimode_rcpsp_calendar_representative(self):
        # put skills as ressource.
        from skdecide.builders.discrete_optimization.rcpsp.rcpsp_model import RCPSPModelCalendar
        if len(self.resources_list) == 0:
            skills_availability = {s: [0] * int(self.horizon)
                                   for s in self.skills_set}
        else:
            skills_availability = {s: [0]*len(self.resources_availability[self.resources_list[0]])
                                   for s in self.skills_set}
        for emp in self.employees:
            for j in range(len(self.employees[emp].calendar_employee)):
                if self.employees[emp].calendar_employee[min(j,
                                                             len(self.employees[emp].calendar_employee)-1)]:
                    for s in self.employees[emp].dict_skill:
                        skills_availability[s][min(j,
                                                   len(skills_availability[s])-1)] += self.employees[emp].dict_skill[s].skill_value
        res_availability = deepcopy(self.resources_availability)
        for s in skills_availability:
            res_availability[s] = [int(x) for x in skills_availability[s]]
        mode_details = deepcopy(self.mode_details)
        for task in mode_details:
            for mode in mode_details[task]:
                for r in self.resources_set:
                    if r not in mode_details[task][mode]:
                        mode_details[task][mode][r] = int(0)
                for s in self.skills_set:
                    if s not in mode_details[task][mode]:
                        mode_details[task][mode][s] = int(0)
        rcpsp_model = RCPSPModelCalendar(resources=res_availability,
                                         non_renewable_resources=list(self.non_renewable_resources),
                                         mode_details=mode_details,
                                         successors=self.successors,
                                         horizon=self.horizon,
                                         horizon_multiplier=self.horizon_multiplier,
                                         name_task={i: str(i) for i in self.tasks},
                                         calendar_details=None)
        return rcpsp_model

    def compute_graph(self)->Graph:
        nodes = [(n, {mode: self.mode_details[n][mode]["duration"]
                      for mode in self.mode_details[n]})
                 for n in range(1, self.n_jobs_non_dummy+3)]
        edges = []
        for n in self.successors:
            for succ in self.successors[n]:
                edges += [(n, succ, {})]
        return Graph(nodes, edges, False)

    # @abstractmethod
    def evaluate_function(self, rcpsp_sol: MS_RCPSPSolution):
        makespan = rcpsp_sol.schedule[self.sink_task]['end_time']
        return makespan

    @abstractmethod
    def evaluate_from_encoding(self, int_vector, encoding_name):
        ...

    def evaluate(self, rcpsp_sol: MS_RCPSPSolution) -> Dict[str, float]:
        obj_makespan = self.evaluate_function(rcpsp_sol)
        return {'makespan': obj_makespan}

    def evaluate_mobj(self, rcpsp_sol: MS_RCPSPSolution):
        return self.evaluate_mobj_from_dict(self.evaluate(rcpsp_sol))

    def evaluate_mobj_from_dict(self, dict_values: Dict[str, float]):
        return TupleFitness(np.array([-dict_values["makespan"]]),
                            1)

    def satisfy(self, rcpsp_sol: MS_RCPSPSolution)->bool:
        # check the skills :
        if len(rcpsp_sol.schedule) != self.nb_tasks:
            return False
        for task in self.tasks:
            mode = rcpsp_sol.modes[task]
            required_skills = {s: self.mode_details[task][mode][s]
                               for s in self.mode_details[task][mode]
                               if s in self.skills_set and self.mode_details[task][mode][s] > 0}
            # Skills for the given task are used
            if len(required_skills) > 0:
                for skill in required_skills:
                    employees_used = [self.employees[emp].dict_skill[skill].skill_value
                                      for emp in rcpsp_sol.employee_usage[task]
                                      if skill in rcpsp_sol.employee_usage[task][emp]]
                    if sum(employees_used) < required_skills[skill]:
                        print('1')
                        return False
            if task in rcpsp_sol.employee_usage:
                employee_used = [emp for emp in rcpsp_sol.employee_usage[task]]
                # print(rcpsp_sol.employee_usage)
                # employee available at this time
                if len(employee_used) > 0:
                    for e in employee_used:
                        if not all(self.employees[e].calendar_employee[t]
                                   for t in range(rcpsp_sol.schedule[task]["start_time"],
                                                  rcpsp_sol.schedule[task]["end_time"])):
                            print("Task : ", task)
                            print("Employee : ", e)
                            print(e, [self.employees[e].calendar_employee[t]
                                      for t in range(rcpsp_sol.schedule[task]["start_time"],
                                                     rcpsp_sol.schedule[task]["end_time"])])
                            print('Problem with employee availability')
                            return False
        overlaps = [(t1, t2) for t1 in self.tasks for t2 in self.tasks
                    if t2>t1 and intersect((rcpsp_sol.schedule[t1]["start_time"],
                                            rcpsp_sol.schedule[t1]["end_time"]),
                                            (rcpsp_sol.schedule[t2]["start_time"],
                                             rcpsp_sol.schedule[t2]["end_time"]))
                    is not None]
        for t1, t2 in overlaps:
            if any(k in rcpsp_sol.employee_usage.get(t2, {})
                   for k in rcpsp_sol.employee_usage.get(t1, {})):
                print("Worker working on 2 task the same time")
                return False
        # ressource usage respected
        makespan = rcpsp_sol.schedule[self.sink_task]['end_time']
        for t in range(makespan):
            resource_usage = {}
            for res in self.resources_set:
                resource_usage[res] = 0
            for act_id in self.tasks:
                start = rcpsp_sol.schedule[act_id]['start_time']
                end = rcpsp_sol.schedule[act_id]['end_time']
                mode = rcpsp_sol.modes[act_id]
                if start <= t and t < end:
                    # print(act_id)
                    for res in self.resources_set:  # self.mode_details[act_id][mode]:
                        # print('res: ', res)
                        # print('adding usage from act', act_id)
                        # print('mode:', mode)
                        # print('self.mode_details[act_id][mode][res]: ', self.mode_details[act_id][mode][res])
                        resource_usage[res] += self.mode_details[act_id][mode][res]
            for res in self.resources_set:
                if resource_usage[res] > self.resources_availability[res][t]:
                    print('Time step resource violation: time: ', t, 'res', res,
                          'res_usage: ', resource_usage[res], 'res_avail: ',
                          self.resources_availability[res][t])
                    print('3')
                    return False

        # Check for non-renewable resource violation
        for res in self.non_renewable_resources:
            usage = 0
            for act_id in self.tasks:
                mode = rcpsp_sol.modes[act_id]
                usage += self.mode_details[act_id][mode][res]
            if usage > self.resources_availability[res][0]:
                print('Non-renewable res', res, 'res_usage: ',
                      usage, 'res_avail: ', self.resources_availability[res][0])
                return False
        # Check precedences / successors
        for act_id in list(self.successors.keys()):
            for succ_id in self.successors[act_id]:
                start_succ = rcpsp_sol.schedule[succ_id]['start_time']
                end_pred = rcpsp_sol.schedule[act_id]['end_time']
                if start_succ < end_pred:
                    print('Precedence relationship broken: ', act_id, 'end at ', end_pred, 'while ', succ_id,
                          'start at', start_succ)
                    return False
        return True

    def __str__(self):
        val = "RCPSP model"
        return val

    def get_solution_type(self):
        return MS_RCPSPSolution

    def get_attribute_register(self) -> EncodingRegister:
        dict_register = {}
        dict_register["modes"] = {"name": "modes",
                                  "type": [Dict[int, int]]}
        # print('max_number_modes: ', max_number_modes)
        dict_register["schedule"] = {"name": "schedule",
                                     "type": [Dict[int, Dict[str, int]]]}
        dict_register["employee_usage"] = {"name": "employee_usage",
                                           "type": [Dict[int, Dict[int, Set[str]]]]}
        return EncodingRegister(dict_register)

    def get_objective_register(self) -> ObjectiveRegister:
        dict_objective = {"makespan": {"type": TypeObjective.OBJECTIVE, "default_weight": -1},
                          "mean_resource_reserve": {"type": TypeObjective.OBJECTIVE, "default_weight": 1}}
        return ObjectiveRegister(objective_sense=ModeOptim.MAXIMIZATION,
                                 objective_handling=ObjectiveHandling.SINGLE,
                                 dict_objective_to_doc=dict_objective)

    def copy(self):
        return MS_RCPSPModel(skills_set=self.skills_set,
                             resources_set=self.resources_set,
                             non_renewable_resources=self.non_renewable_resources,
                             resources_availability=self.resources_availability,
                             employees=self.employees,
                             employees_availability=self.employees_availability,
                             mode_details=self.mode_details,
                             successors=self.successors,
                             horizon=self.horizon,
                             horizon_multiplier=self.horizon_multiplier)

    def to_variant_model(self):
        return MS_RCPSPModel_Variant(skills_set=self.skills_set,
                                     resources_set=self.resources_set,
                                     non_renewable_resources=self.non_renewable_resources,
                                     resources_availability=self.resources_availability,
                                     employees=self.employees,
                                     employees_availability=self.employees_availability,
                                     mode_details=self.mode_details,
                                     successors=self.successors,
                                     horizon=self.horizon,
                                     horizon_multiplier=self.horizon_multiplier,
                                     one_unit_per_task_max=self.one_unit_per_task_max)

    def get_dummy_solution(self):
        # TODO
        return None


class MS_RCPSPModel_Variant(MS_RCPSPModel):

    def __init__(self,
                 skills_set: Set[str],
                 resources_set: Set[str],
                 non_renewable_resources: Set[str],
                 resources_availability:  Dict[str, List[int]],
                 employees: Dict[int,
                                 Employee],
                 employees_availability: List[int],
                 mode_details: Dict[int, Dict[int, Dict[str, int]]],
                 successors: Dict[int, List[int]],
                 horizon,
                 horizon_multiplier=1,
                 sink_task: Optional[int]=None,
                 source_task: Optional[int]=None,
                 one_unit_per_task_max: bool=False):
        MS_RCPSPModel.__init__(self, skills_set=skills_set,
                             resources_set=resources_set,
                             non_renewable_resources=non_renewable_resources,
                             resources_availability=resources_availability,
                             employees=employees,
                             employees_availability=employees_availability,
                             mode_details=mode_details,
                             successors=successors,
                             horizon=horizon,
                             horizon_multiplier=horizon_multiplier)
        self.fixed_modes = None
        self.fixed_permutation = None
        self.fixed_priority_worker_per_task = None

    def get_attribute_register(self) -> EncodingRegister:
        dict_register = {}
        max_number_modes = max([len(list(self.mode_details[x].keys())) for x in self.mode_details.keys()])
        dict_register["priority_list_task"] = {"name": "priority_list_task",
                                               "type": [TypeAttribute.PERMUTATION,
                                                        TypeAttribute.PERMUTATION_RCPSP],
                                               "range": range(self.n_jobs_non_dummy),
                                               "n": self.n_jobs_non_dummy}
        dict_register["priority_worker_per_task_perm"] = {"name": "priority_worker_per_task",
                                                          "type": [TypeAttribute.PERMUTATION,
                                                                   TypeAttribute.PERMUTATION_RCPSP],
                                                          "range": range(self.n_jobs_non_dummy*len(self.employees.keys())),
                                                          "n": self.n_jobs_non_dummy*len(self.employees.keys())}
        dict_register["priority_worker_per_task"] = {"name": "priority_worker_per_task",
                                                     "type": [List[List[int]]]}
        dict_register["modes_vector"] = {"name": "modes_vector",
                                         "n": self.n_jobs_non_dummy,
                                         "arrity": max_number_modes,
                                         "type": [TypeAttribute.LIST_INTEGER]}

        mode_arrity = [len(list(self.mode_details[list(self.mode_details.keys())[i]].keys()))
                       for i in range(1, len(self.mode_details.keys()) - 1)]
        dict_register["modes_arrity_fix"] = {"name": "modes_vector",
                                             "type": [TypeAttribute.LIST_INTEGER_SPECIFIC_ARRITY],
                                             "n": self.n_jobs_non_dummy,
                                             "arrities": mode_arrity}
        dict_register["modes_arrity_fix_from_0"] = {"name": "modes_vector_from0",
                                                    "type": [TypeAttribute.LIST_INTEGER_SPECIFIC_ARRITY],
                                                    "n": self.n_jobs_non_dummy,
                                                    "arrities": mode_arrity}

        dict_register["schedule"] = {"name": "schedule",
                                     "type": [Dict[int, Dict[str, int]]]}
        dict_register["employee_usage"] = {"name": "employee_usage",
                                           "type": [Dict[int, Dict[int, Set[str]]]]}
        return EncodingRegister(dict_register)

    def get_dummy_solution(self):
        # TODO
        return MS_RCPSPSolution_Variant(problem=self,
                                        priority_list_task=[i for i in range(self.n_jobs_non_dummy)],
                                        modes_vector=[1 for i in range(self.n_jobs_non_dummy)],
                                        priority_worker_per_task=[[w for w in self.employees]
                                                                  for i in range(self.n_jobs_non_dummy)])

    def evaluate_function(self, rcpsp_sol: MS_RCPSPSolution_Variant):
        try:
            if rcpsp_sol._schedule_to_recompute:
                rcpsp_sol.do_recompute()
        except:
            pass
        return super().evaluate_function(rcpsp_sol)

    def set_fixed_attributes(self, encoding_str: str, sol: MS_RCPSPSolution_Variant):
        att = self.get_attribute_register().dict_attribute_to_type[encoding_str]['name']
        if att == 'modes_vector':
            self.set_fixed_modes(sol.modes_vector)
            print('self.fixed_modes:', self.fixed_modes)
        elif att == 'modes_vector_from0':
            modes_corrected = [x + 1 for x in sol.modes_vector]
            self.set_fixed_modes(modes_corrected)
            print('self.fixed_modes:', self.fixed_modes)
        elif att == 'priority_worker_per_task':
            self.set_fixed_priority_worker_per_task(sol.priority_worker_per_task)
            print('self.fixed_priority_worker_per_task:', self.fixed_priority_worker_per_task)
        elif att == 'priority_list_task':
            self.set_fixed_task_permutation(sol.priority_list_task)
            print('self.fixed_permutation:', self.fixed_permutation)

    def set_fixed_modes(self, fixed_modes):
        self.fixed_modes = fixed_modes

    def set_fixed_task_permutation(self, fixed_permutation):
        self.fixed_permutation = fixed_permutation

    def set_fixed_priority_worker_per_task(self, fixed_priority_worker_per_task):
        self.fixed_priority_worker_per_task = fixed_priority_worker_per_task

    def set_fixed_priority_worker_per_task_from_permutation(self, permutation):
        self.fixed_priority_worker_per_task = self.convert_fixed_priority_worker_per_task_from_permutation(permutation)

    def convert_fixed_priority_worker_per_task_from_permutation(self, permutation):
        priority_worker_per_task_corrected = []
        for i in range(self.n_jobs_non_dummy):
            tmp = []
            for j in range(len(self.employees.keys())):
                tmp.append(permutation[i * len(self.employees.keys()) + j])
            tmp_corrected = [int(x) for x in ss.rankdata(tmp)]
            priority_worker_per_task_corrected.append(tmp_corrected)
        return priority_worker_per_task_corrected

    def evaluate_from_encoding(self, int_vector, encoding_name):
        if encoding_name == 'priority_list_task':
            # change the permutation in the solution with int_vector and set the modes with self.fixed_modes
            rcpsp_sol = MS_RCPSPSolution_Variant(problem=self,
                                                 priority_list_task=int_vector,
                                                 modes_vector=self.fixed_modes,
                                                 priority_worker_per_task=self.fixed_priority_worker_per_task)
        elif encoding_name == 'modes_vector':
            # change the modes in the solution with int_vector and set the permutation with self.fixed_permutation
            modes_corrected = int_vector
            rcpsp_sol = MS_RCPSPSolution_Variant(problem=self,
                                                 priority_list_task=self.fixed_permutation,
                                                 modes_vector=modes_corrected,
                                                 priority_worker_per_task=self.fixed_priority_worker_per_task)
        elif encoding_name == 'modes_vector_from0':
            # change the modes in the solution with int_vector and set the permutation with self.fixed_permutation
            modes_corrected = [x+1 for x in int_vector]
            rcpsp_sol = MS_RCPSPSolution_Variant(problem=self,
                                                 priority_list_task=self.fixed_permutation,
                                                 modes_vector=modes_corrected,
                                                 priority_worker_per_task=self.fixed_priority_worker_per_task)
        elif encoding_name == 'priority_worker_per_task':
            # change the resource permutation priority lists in the solution from int_vector and set the permutation
            # with self.fixed_permutation and the modes with self.fixed_modes
            priority_worker_per_task_corrected = []
            for i in range(self.n_jobs_non_dummy):
                tmp = []
                for j in range(len(self.employees.keys())):
                    tmp.append(int_vector[i*len(self.employees.keys())+j])
                tmp_corrected = [int(x) for x in ss.rankdata(tmp)]
                priority_worker_per_task_corrected.append(tmp_corrected)
            rcpsp_sol = MS_RCPSPSolution_Variant(problem=self,
                                                 priority_list_task=self.fixed_permutation,
                                                 modes_vector=self.fixed_modes,
                                                 priority_worker_per_task=priority_worker_per_task_corrected)
        objectives = self.evaluate(rcpsp_sol)
        return objectives

    def get_solution_type(self):
        return MS_RCPSPSolution_Variant
