import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../"))
from skdecide.builders.discrete_optimization.rcpsp.rcpsp_model import RCPSPModel, SingleModeRCPSPModel, \
    MultiModeRCPSPModel, RCPSPSolution
import os
import sys
from skdecide.builders.discrete_optimization.generic_tools.path_tools import abspath_from_file
path_to_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/rcpsp/")
files_available = [os.path.join(path_to_data, f) for f in os.listdir(path_to_data)]
# path_to_results = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/rcpsp_sols/")
# results_available = [os.path.join(path_to_results, f) for f in os.listdir(path_to_results)]


def get_data_available():
    files = [f for f in os.listdir(path_to_data) if "pk" not in f and "json" not in f]
    return [os.path.join(path_to_data, f) for f in files]


def parse_psplib(input_data):
    # parse the input
    # print('input_data\n',input_data)
    lines = input_data.split('\n')

    # Retrieving section bounds
    horizon_ref_line_index = lines.index('RESOURCES')-1

    prec_ref_line_index = lines.index('PRECEDENCE RELATIONS:')
    prec_start_line_index = prec_ref_line_index + 2
    # print('prec_start_line_index: ', prec_start_line_index)
    duration_ref_line_index = lines.index('REQUESTS/DURATIONS:')
    prec_end_line_index = duration_ref_line_index - 2
    duration_start_line_index = duration_ref_line_index + 3
    # print('duration_start_line_index: ', duration_start_line_index)
    res_ref_line_index = lines.index('RESOURCEAVAILABILITIES:')
    #res_ref_line_index = lines.index('RESOURCE AVAILABILITIES')
    duration_end_line_index = res_ref_line_index - 2
    res_start_line_index = res_ref_line_index + 1
    # print('res_start_line_index: ', res_start_line_index)

    # print('prec_end_line_index: ', prec_end_line_index)
    # print('duration_end_line_index: ', duration_end_line_index)

    # Parsing horizon
    tmp = lines[horizon_ref_line_index].split()
    horizon = int(tmp[2])

    # Parsing resource information
    tmp1 = lines[res_start_line_index].split()
    tmp2 = lines[res_start_line_index+1].split()
    resources = {str(tmp1[(i*2)])+str(tmp1[(i*2)+1]): int(tmp2[i]) for i in range(len(tmp2))}
    non_renewable_resources = [name for name in list(resources.keys()) if name.startswith('N')]
    n_resources = len(resources.keys())

    # Parsing precedence relationship
    multi_mode = False
    successors = {}
    for i in range(prec_start_line_index, prec_end_line_index+1):
        tmp = lines[i].split()
        task_id = int(tmp[0])
        n_modes = int(tmp[1])
        n_successors = int(tmp[2])
        successors[task_id] = [int(x) for x in tmp[3:(3+n_successors)]]

    # Parsing mode and duration information
    mode_details = {}
    for i in range(duration_start_line_index, duration_end_line_index + 1):
        tmp = lines[i].split()
        if len(tmp) == 3+n_resources:
            task_id = int(tmp[0])
            mode_id = int(tmp[1])
            duration = int(tmp[2])
            resources_usage = [int(x) for x in tmp[3:(3+n_resources)]]
        else:
            multi_mode = True
            mode_id = int(tmp[0])
            duration = int(tmp[1])
            # resources_usage = tmp[2:(3 + n_resources)]
            resources_usage = [int(x) for x in tmp[2:(3 + n_resources)]]

        if int(task_id) not in list(mode_details.keys()):
            mode_details[int(task_id)] = {}
        mode_details[int(task_id)][mode_id] = {}  #Dict[int, Dict[str, int]]
        mode_details[int(task_id)][mode_id]['duration'] = duration
        for i in range(n_resources):
            mode_details[int(task_id)][mode_id][list(resources.keys())[i]] = resources_usage[i]

    if multi_mode:
        problem = MultiModeRCPSPModel(resources=resources,
                                      non_renewable_resources=non_renewable_resources,
                                      mode_details=mode_details,
                                      successors=successors,
                                      horizon=horizon,
                                      horizon_multiplier=30)
    else:
        problem = SingleModeRCPSPModel(resources=resources,
                                       non_renewable_resources=non_renewable_resources,
                                       mode_details=mode_details,
                                       successors=successors,
                                       horizon=horizon,
                                       horizon_multiplier=30)
    return problem


def parse_file(file_path)->RCPSPModel:
    with open(file_path, 'r') as input_data_file:
        input_data = input_data_file.read()
        rcpsp_model = parse_psplib(input_data)
        return rcpsp_model



