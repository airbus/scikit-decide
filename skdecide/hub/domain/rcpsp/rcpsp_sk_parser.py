# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

# Load rcpsp domains from psplib files.
# You need the discrete optimisation library to be able to use those.
from typing import Union

from skdecide.hub.domain.rcpsp.rcpsp_sk import MSRCPSP


def load_domain(file_path):
    """"""
    from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel
    from discrete_optimization.rcpsp.rcpsp_parser import parse_file

    from skdecide.hub.domain.rcpsp.rcpsp_sk import MRCPSP, RCPSP

    rcpsp_model: RCPSPModel = parse_file(file_path)
    if not rcpsp_model.is_rcpsp_multimode():
        my_domain = RCPSP(
            resource_names=rcpsp_model.resources_list,
            task_ids=sorted(rcpsp_model.mode_details.keys()),
            tasks_mode=rcpsp_model.mode_details,
            successors=rcpsp_model.successors,
            max_horizon=rcpsp_model.horizon,
            resource_availability=rcpsp_model.resources,
            resource_renewable={
                r: r not in rcpsp_model.non_renewable_resources
                for r in rcpsp_model.resources_list
            },
        )
    else:
        my_domain = MRCPSP(
            resource_names=rcpsp_model.resources_list,
            task_ids=sorted(rcpsp_model.mode_details.keys()),
            tasks_mode=rcpsp_model.mode_details,
            successors=rcpsp_model.successors,
            max_horizon=rcpsp_model.horizon,
            resource_availability=rcpsp_model.resources,
            resource_renewable={
                r: r not in rcpsp_model.non_renewable_resources
                for r in rcpsp_model.resources_list
            },
        )

    return my_domain


def load_multiskill_domain(file_path):
    from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill_parser import (
        parse_file,
    )

    model_msrcpsp, new_tame_to_original_task_id = parse_file(
        file_path, max_horizon=2000
    )
    resource_type_names = list(model_msrcpsp.resources_list)
    resource_skills = {r: {} for r in resource_type_names}
    resource_availability = {
        r: model_msrcpsp.resources_availability[r][0]
        for r in model_msrcpsp.resources_availability
    }
    resource_renewable = {
        r: r not in model_msrcpsp.non_renewable_resources
        for r in model_msrcpsp.resources_list
    }
    resource_unit_names = []
    for employee in model_msrcpsp.employees:
        resource_unit_names += ["employee-" + str(employee)]
        resource_skills[resource_unit_names[-1]] = {}
        resource_availability[resource_unit_names[-1]] = 1
        resource_renewable[resource_unit_names[-1]] = True
        for s in model_msrcpsp.employees[employee].dict_skill:
            resource_skills[resource_unit_names[-1]][s] = (
                model_msrcpsp.employees[employee].dict_skill[s].skill_value
            )

    return MSRCPSP(
        skills_names=list(model_msrcpsp.skills_set),
        resource_unit_names=resource_unit_names,
        resource_type_names=resource_type_names,
        resource_skills=resource_skills,
        task_ids=sorted(model_msrcpsp.mode_details.keys()),
        tasks_mode=model_msrcpsp.mode_details,
        successors=model_msrcpsp.successors,
        max_horizon=model_msrcpsp.horizon,
        resource_availability=resource_availability,
        resource_renewable=resource_renewable,
    )
