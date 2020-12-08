# Load rcpsp domains from psplib files.
# You need the discrete optimisation library to be able to use those.
from typing import Union
from skdecide.hub.domain.rcpsp.rcpsp_sk import RCPSP, MRCPSP, MSRCPSP


def load_domain(file_name="j1201_1.sm"):
    from skdecide.builders.discrete_optimization.rcpsp.rcpsp_parser import parse_file, \
        get_data_available, SingleModeRCPSPModel, MultiModeRCPSPModel
    from skdecide.hub.domain.rcpsp.rcpsp_sk import RCPSP, MRCPSP
    files = get_data_available()
    # print(files)
    files = [f for f in files if file_name in f]  # Single mode RCPSP
    # files = [f for f in files if 'j1010_5.mm' in f]  # Multi mode RCPSP
    file_path = files[0]
    rcpsp_model: Union[SingleModeRCPSPModel, MultiModeRCPSPModel] = parse_file(file_path)
    if isinstance(rcpsp_model, SingleModeRCPSPModel):
        my_domain = RCPSP(resource_names=rcpsp_model.resources_list,
                          task_ids=sorted(rcpsp_model.mode_details.keys()),
                          tasks_mode=rcpsp_model.mode_details,
                          successors=rcpsp_model.successors,
                          max_horizon=rcpsp_model.horizon,
                          resource_availability=rcpsp_model.resources,
                          resource_renewable={r: r not in rcpsp_model.non_renewable_resources
                                               for r in rcpsp_model.resources_list})
    elif isinstance(rcpsp_model, MultiModeRCPSPModel):
        my_domain = MRCPSP(resource_names=rcpsp_model.resources_list,
                           task_ids=sorted(rcpsp_model.mode_details.keys()),
                           tasks_mode=rcpsp_model.mode_details,
                           successors=rcpsp_model.successors,
                           max_horizon=rcpsp_model.horizon,
                           resource_availability=rcpsp_model.resources,
                           resource_renewable={r: r not in rcpsp_model.non_renewable_resources
                                               for r in rcpsp_model.resources_list})

    return my_domain


def load_multiskill_domain():
    from skdecide.builders.discrete_optimization.rcpsp_multiskill.rcpsp_multiskill_parser import parse_file, \
        get_data_available, MS_RCPSPModel
    model_msrcpsp, new_tame_to_original_task_id = parse_file(get_data_available()[0],
                                                             max_horizon=2000)
    resource_type_names = list(model_msrcpsp.resources_list)
    resource_skills = {r: {} for r in resource_type_names}
    resource_availability = {r: model_msrcpsp.resources_availability[r][0]
                             for r in model_msrcpsp.resources_availability}
    resource_renewable = {r: r not in model_msrcpsp.non_renewable_resources
                          for r in model_msrcpsp.resources_list}
    resource_unit_names = []
    for employee in model_msrcpsp.employees:
        resource_unit_names += ["employee-"+str(employee)]
        resource_skills[resource_unit_names[-1]] = {}
        resource_availability[resource_unit_names[-1]] = 1
        resource_renewable[resource_unit_names[-1]] = True
        for s in model_msrcpsp.employees[employee].dict_skill:
            resource_skills[resource_unit_names[-1]][s] = model_msrcpsp.employees[employee].dict_skill[s].skill_value

    return MSRCPSP(skills_names=list(model_msrcpsp.skills_set),
                   resource_unit_names=resource_unit_names,
                   resource_type_names=resource_type_names,
                   resource_skills=resource_skills,
                   task_ids=sorted(model_msrcpsp.mode_details.keys()),
                   tasks_mode=model_msrcpsp.mode_details,
                   successors=model_msrcpsp.successors,
                   max_horizon=model_msrcpsp.horizon,
                   resource_availability=resource_availability,
                   resource_renewable=resource_renewable)
