import os, sys
from typing import Tuple, Dict

from skdecide.builders.discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import MS_RCPSPSolution_Variant, \
    MS_RCPSPModel, MS_RCPSPModel_Variant, Employee, SkillDetail, MS_RCPSPSolution
import os
path_to_data =\
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/rcpsp_multiskill/dataset_def/")


def get_data_available():
    files = [f for f in os.listdir(path_to_data) if "pk" not in f and "json" not in f]
    return [os.path.join(path_to_data, f) for f in files]


def get_results_directory(directory):
    files = [f for f in os.listdir(directory) if ".sol" in f]
    return [os.path.join(directory, f) for f in files]


def parse_imopse(input_data, max_horizon=None):
    # parse the input
    # print('input_data\n',input_data)
    lines = input_data.split('\n')

    # "General characteristics:
    #     Tasks: 161
    #     Resources: 10
    #     Precedence relations: 321
    #     Number of skill types: 9
    #     ====================================================================================================================
    #     ResourceID 	 Salary 	 Skills
    #     1	 	 	14.2	 	 Q2: 0 	  Q3: 2 	  Q1: 0 	  Q4: 2 	  Q7: 1 	  Q8: 2
    #     2	 	 	31.2	 	 Q0: 0 	  Q4: 2 	  Q7: 1 	  Q3: 1 	  Q8: 2 	  Q2: 0
    #     3	 	 	34.4	 	 Q4: 0 	  Q2: 1 	  Q6: 2 	  Q3: 1 	  Q0: 1 	  Q5: 0
    #     4	 	 	26.0	 	 Q5: 2 	  Q1: 1 	  Q4: 1 	  Q8: 2 	  Q0: 2 	  Q2: 2
    #     5	 	 	30.8	 	 Q8: 0 	  Q7: 1 	  Q3: 1 	  Q1: 2 	  Q4: 1 	  Q5: 1
    #     6	 	 	17.3	 	 Q6: 1 	  Q3: 2 	  Q4: 2 	  Q2: 0 	  Q7: 2 	  Q1: 0
    #     7	 	 	19.8	 	 Q1: 2 	  Q4: 2 	  Q5: 0 	  Q7: 1 	  Q3: 1 	  Q6: 2
    #     8	 	 	35.8	 	 Q2: 1 	  Q0: 1 	  Q3: 2 	  Q6: 0 	  Q7: 0 	  Q8: 1
    #     9	 	 	37.6	 	 Q7: 0 	  Q5: 2 	  Q2: 0 	  Q1: 0 	  Q0: 1 	  Q3: 1
    #     10	 	 	23.5	 	 Q8: 1 	  Q5: 1 	  Q1: 2 	  Q6: 0 	  Q4: 0 	  Q3: 2 	 "
    nb_task = None
    nb_worker = None
    nb_precedence_relation = None
    nb_skills = None
    resource_zone = False
    task_zone = False
    resource_dict = {}
    task_dict = {}
    real_skills_found = set()
    for line in lines:
        words = line.split()
        if len(words) == 2 and words[0] == "Tasks:":
            nb_task = int(words[1])
            continue
        if len(words) == 2 and words[0] == "Resources:":
            nb_worker = int(words[1])
            continue
        if len(words) == 3 and words[0] == "Precedence" and words[1] == "relations:":
            nb_precedence_relation = int(words[2])
            continue
        if len(words) == 5 and words[0] == "Number" and words[1] == "of":
            nb_skills = int(words[4])
            continue
        if len(words) == 0:
            continue
        if words[0] == "ResourceID":
            resource_zone = True
            continue
        if words[0] == "TaskID":
            task_zone = True
            continue
        if resource_zone:
            if words[0][0] == "=":
                resource_zone = False
                continue
            else:
                id_worker = words[0]
                resource_dict[id_worker] = {"salary": float(words[1])}
                for word in words[2:]:
                    if word[0] == "Q":
                        current_skill = word[:-1]
                        continue
                    resource_dict[id_worker][current_skill] = int(word)+1
                    real_skills_found.add(current_skill)
        if task_zone:
            if words[0][0] == "=":
                task_zone = False
                continue
            else:
                task_id = int(words[0])
                if task_id not in task_dict:
                    task_dict[task_id] = {"id": task_id, "successors": [], "skills": {}}
                task_dict[task_id]["duration"] = int(words[1])
            i = 2
            while i < len(words):
                if words[i][0] == "Q":
                    current_skill = words[i][:-1]
                    task_dict[task_id]["skills"][current_skill] = int(words[i+1])+1
                    real_skills_found.add(current_skill)
                    i = i+2
                    continue
                else:
                    if "precedence" not in task_dict[task_id]:
                        task_dict[task_id]["precedence"] = []
                    task_dict[task_id]["precedence"] += [int(words[i])]
                    if int(words[i]) not in task_dict:
                        task_dict[int(words[i])] = {"id": int(words[i]), "successors": [], "skills": {}}
                    if "successors" not in task_dict[int(words[i])]:
                        task_dict[int(words[i])]["successors"] = []
                    task_dict[int(words[i])]["successors"] += [task_id]
                    i += 1
    # print(resource_dict)
    # print(task_dict)
    sorted_task_names = sorted(task_dict.keys())
    task_id_to_new_name = {sorted_task_names[i]: i+2 for i in range(len(sorted_task_names))}
    new_tame_to_original_task_id = {task_id_to_new_name[ind]: ind for ind in task_id_to_new_name}
    mode_details = {task_id_to_new_name[task_id]:
                    {1: {"duration": task_dict[task_id]["duration"]}}
                    for task_id in task_dict}
    resource_dict = {int(i): resource_dict[i] for i in resource_dict}
    # skills = set(["Q"+str(i) for i in range(nb_skills)])
    skills = real_skills_found
    for task_id in task_dict:
        for skill in skills:
            req_squill = task_dict[task_id]["skills"].get(skill, 0.)
            mode_details[task_id_to_new_name[task_id]][1][skill] = req_squill
    mode_details[1] = {1: {"duration": 0}}
    for skill in skills:
        mode_details[1][1][skill] = int(0)
    max_t = max(mode_details)
    mode_details[max_t+1] = {1: {"duration": 0}}
    for skill in skills:
        mode_details[max_t+1][1][skill] = int(0)
    successors = {task_id_to_new_name[task_id]:
                      [task_id_to_new_name[t]
                       for t in task_dict[task_id]["successors"]]+[max_t+1]
                  for task_id in task_dict}
    successors[max_t+1] = []
    successors[1] = [k for k in successors]
    # max_horizon = 2*sum([task_dict[task_id]["duration"] for task_id in task_dict])
    max_horizon = 300 if max_horizon is None else max_horizon
    return MS_RCPSPModel(skills_set=set(real_skills_found),
                         resources_set=set(),
                         non_renewable_resources=set(),
                         resources_availability={},
                         employees={res: Employee(dict_skill={skill: SkillDetail(skill_value=resource_dict[res][skill],
                                                                                 efficiency_ratio=1., experience=1.)
                                                              for skill in resource_dict[res] if skill != "salary"},
                                                  salary=resource_dict[res]["salary"],
                                                  calendar_employee=[True]*max_horizon)
                                    for res in resource_dict},
                         employees_availability=[len(resource_dict)]*max_horizon,
                         mode_details=mode_details,
                         successors=successors,
                         horizon=max_horizon,
                         source_task=1,
                         sink_task=max_t+1, one_unit_per_task_max=True), new_tame_to_original_task_id


def parse_results(input_data):
    #if isinstance(input_data, str):
    #    with open(input_data, 'r') as input_data_file:
    #        return parse_results(input_data_file.read())
    lines = input_data.split('\n')
    schedule = {}
    assignation = {}
    for i in range(1, len(lines)):
        line = lines[i]
        words = line.split()
        try:
            hour = int(words[0])
            for j in range(1, len(words)):
                w = words[j].split("-")
                ressource_id = w[0]
                task_id = int(w[1])
                assignation[task_id] = ressource_id
                schedule[task_id] = hour
        except:
            pass
    return schedule, assignation


def recompute_solution(ms_rcpsp_model: MS_RCPSPModel,
                       new_name_to_original_task_id: Dict,
                       schedule: Dict,
                       assignation: Dict):
    reverse_name = {new_name_to_original_task_id[key]: key
                    for key in new_name_to_original_task_id}
    schedule_recomputed = {}
    modes = {}
    employee_usage = {}
    for task in schedule:
        schedule_recomputed[reverse_name[task]] = {"start_time": schedule[task]}
        duration = ms_rcpsp_model.mode_details[reverse_name[task]][1]["duration"]
        schedule_recomputed[reverse_name[task]]["end_time"] = schedule_recomputed[reverse_name[task]]["start_time"]\
                                                              + duration
        modes[reverse_name[task]] = 1
        employee = assignation[task]
        skills_used_by_employees = [s for s in ms_rcpsp_model.employees[employee].dict_skill
                                    if ms_rcpsp_model.employees[employee].dict_skill[s].skill_value > 0
                                    and s in ms_rcpsp_model.mode_details[reverse_name[task]][1]
                                    and ms_rcpsp_model.mode_details[reverse_name[task]][1][s] > 0]
        employee_usage[reverse_name[task]] = {employee: set(skills_used_by_employees)}
    employee_usage[ms_rcpsp_model.source_task] = {}
    employee_usage[ms_rcpsp_model.sink_task] = {}
    modes[ms_rcpsp_model.source_task] = 1
    modes[ms_rcpsp_model.sink_task] = 1
    schedule_recomputed[ms_rcpsp_model.source_task] = {"start_time": 0, "end_time": 0}
    last_time = max([schedule_recomputed[x]["end_time"] for x in schedule_recomputed])
    schedule_recomputed[ms_rcpsp_model.sink_task] = {"start_time": last_time,
                                                     "end_time": last_time}
    return MS_RCPSPSolution(problem=ms_rcpsp_model,
                            modes=modes,
                            schedule=schedule_recomputed,
                            employee_usage=employee_usage)


def parse_file(file_path, max_horizon=None)->Tuple[MS_RCPSPModel, Dict]:
    with open(file_path, 'r') as input_data_file:
        input_data = input_data_file.read()
        rcpsp_model, new_tame_to_original_task_id = parse_imopse(input_data, max_horizon)
        return rcpsp_model, new_tame_to_original_task_id


def parse_one_file():
    files = get_data_available()
    file = [f for f in files if "100_5_20_9_D3.def" in f][0]
    model = parse_file(file)
    print(model)


def write_solution(solution: MS_RCPSPSolution,
                   new_tame_to_original_task_id,
                   file_path=""):
    file1 = open(file_path, "w")
    file1.writelines(["Hour 	 Resource assignments (resource ID - task ID) \n"])
    sorted_task_per_hour = sorted(solution.schedule, key=lambda x: solution.schedule[x]["start_time"])
    strings_hours = {}
    for task in sorted_task_per_hour:
        if task in new_tame_to_original_task_id:
            original_task = new_tame_to_original_task_id[task]
            employees_used = list(solution.employee_usage[task].keys())
            hour = solution.schedule[task]["start_time"]
            if hour not in strings_hours:
                strings_hours[hour] = str(hour)+" "
            for emp in employees_used:
                strings_hours[hour] += str(emp)+"-"+str(original_task)+" "
    file1.writelines([strings_hours[hour]+'\n' for hour in sorted(strings_hours)])
    file1.close()


def run_recompute():
    schedule, assignation = parse_results(open("/Users/poveda_g/Documents/discrete-optimisation/"
                                               "discrete_optimization/data/rcpsp_multiskill/de_best/100_5_20_9_D3.def.sol", "r").read())
    model, name = parse_imopse(open("/Users/poveda_g/Documents/discrete-optimisation/"
                                    "discrete_optimization/data/rcpsp_multiskill/dataset_def/100_5_20_9_D3.def",
                                    "r").read(), max_horizon=1000)
    solution = recompute_solution(model, name, schedule, assignation)
    print(model.evaluate(solution))
    print(model.satisfy(solution))


