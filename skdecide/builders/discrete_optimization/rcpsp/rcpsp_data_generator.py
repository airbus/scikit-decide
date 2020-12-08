from skdecide.builders.discrete_optimization.rcpsp.rcpsp_model import RCPSPModel, RCPSP_H_Model, SingleModeRCPSPModel, MultiModeRCPSPModel, MRCPSP_H_Model
import numpy as np
import random


def generate_rcpsp_with_helper_tasks_data(rcpsp_model: RCPSPModel, original_duration_multiplier,
                                          n_assisted_activities, n_assistants,
                                          probability_of_cross_helper_precedences,
                                          fixed_helper_duration, random_seed):

    np.random.seed(random_seed)
    the_model = rcpsp_model.copy()
    print('n_jobs: ', the_model.n_jobs)
    print('original_duration_multiplier: ', original_duration_multiplier)
    print('n_assisted_activities: ', n_assisted_activities)
    print('n_assistants: ', n_assistants)

    # Add new resource (assistant) to resource types
    n_renewables = len(list(the_model.resources.keys())) - len(the_model.non_renewable_resources)
    print('n_renewables: ', n_renewables)
    additional_resource_str = 'R'+str(n_renewables+1)
    the_model.resources[additional_resource_str] = n_assistants
    print('the_model.resources', the_model.resources)

    # Multiply the original durations (because original durations are quite short and we want helper durations to be even shorter)
    for act_id in the_model.mode_details.keys():
        for mode_id in the_model.mode_details[act_id].keys():
            the_model.mode_details[act_id][mode_id]['duration'] *= original_duration_multiplier

    # Pick randomly activities that will require assistance
    sorted_activities = list(sorted(the_model.mode_details.keys()))
    non_dummy_act_ids = sorted_activities[1:-1]
    assisted_activities = np.random.permutation(non_dummy_act_ids)[:n_assisted_activities].tolist()
    print('assisted_activities picked randomly: ', assisted_activities)

    # Change the id of the last dummy activity (sink) so that it equals to n_jobs+2+n_assisted_activities *2
    # Change it in mode_details, in successors and also change n_jobs
    n_new_activities = n_assisted_activities*2
    old_sink_activity_id = sorted_activities[-1]
    new_sink_activity_id = old_sink_activity_id + n_new_activities
    print('old_sink_activity_id: ', old_sink_activity_id)
    print('new_sink_activity_id: ', new_sink_activity_id)

    temp = the_model.mode_details[old_sink_activity_id].copy()
    del the_model.mode_details[old_sink_activity_id]
    the_model.mode_details[new_sink_activity_id] = temp
    print('updated-1 the_model.mode_details: ', the_model.mode_details)
    the_model.n_jobs += n_new_activities
    print('updated - n_jobs: ', the_model.n_jobs)
    print('original - the_model.successors: ', the_model.successors)
    for act_id in the_model.successors.keys():
        if old_sink_activity_id in the_model.successors[act_id]:
            del the_model.successors[act_id][the_model.successors[act_id].index(old_sink_activity_id)]
            the_model.successors[act_id].append(new_sink_activity_id)
    del the_model.successors[old_sink_activity_id]
    the_model.successors[new_sink_activity_id] = []
    print('updated - the_model.successors: ', the_model.successors)

    # Add additional resource need for original task in mode_details (always 0)
    for act_id in the_model.mode_details.keys():
        for mode_id in  the_model.mode_details[act_id].keys():
            the_model.mode_details[act_id][mode_id][additional_resource_str] = 0
    print('updated-2 the_model.mode_details: ', the_model.mode_details)

    # Add helper activities to mode details:
    for i in range(n_new_activities):
        the_model.mode_details[old_sink_activity_id+i] = {}
        the_model.mode_details[old_sink_activity_id + i][1] = {}
        the_model.mode_details[old_sink_activity_id + i][1]['duration'] = fixed_helper_duration # TODO: Change this fixed duration
        for res_key in the_model.resources.keys():
            if res_key != additional_resource_str:
                the_model.mode_details[old_sink_activity_id + i][1][res_key] = 0
            else:
                the_model.mode_details[old_sink_activity_id + i][1][res_key] = 1
    print('updated-3 the_model.mode_details: ', the_model.mode_details)

    # Add successor relationship in successors
    # + Check that post_helper has the sink activity in case the original activity has it as successor

    pre_helper_activities = {}
    post_helper_activities = {}
    for i in range(n_assisted_activities):
        the_model.successors[old_sink_activity_id + i] = [assisted_activities[i]]
        # original activity as successor of pre-helper
        # pre_helper_activities[old_sink_activity_id + i] = [assisted_activities[i]]
        pre_helper_activities[assisted_activities[i]] = [old_sink_activity_id + i]

        the_model.successors[assisted_activities[i]].append((old_sink_activity_id + n_assisted_activities + i))
        # post-helper as successor of original activity
        the_model.successors[(old_sink_activity_id + n_assisted_activities + i)] = []
        # post_helper_activities[(old_sink_activity_id + n_assisted_activities + i)] = [assisted_activities[i]]
        post_helper_activities[assisted_activities[i]] = [(old_sink_activity_id + n_assisted_activities + i)]

        if new_sink_activity_id in the_model.successors[assisted_activities[i]]:
            print('change in successor sink for id: ', assisted_activities[i])
            del the_model.successors[assisted_activities[i]][the_model.successors[assisted_activities[i]].index(new_sink_activity_id)]
            the_model.successors[(old_sink_activity_id + n_assisted_activities + i)].append(new_sink_activity_id)

    print('the_model.successors:', the_model.successors)
    print('pre_helper_activities: ', pre_helper_activities)
    print('post_helper_activities: ', post_helper_activities)
    # Add precendences between helper activities (e.g. a post_helper required before a pre_helper can represent bringing back tools to main storage for them to be used for another task
    assisted_activities.sort()
    for main_act_id in assisted_activities:
        the_index = assisted_activities.index(main_act_id)
        for main_act_id_2 in assisted_activities[the_index+1:]:
            if random.random() < probability_of_cross_helper_precedences:
                post_act_id = post_helper_activities[main_act_id][0]
                pre_act_id = pre_helper_activities[main_act_id_2][0]
                print('adding cross helper precedence between helper activities: ', post_act_id, 'and', pre_act_id)
                the_model.successors[post_act_id].append(pre_act_id)

    # return the_model
    if isinstance(the_model, SingleModeRCPSPModel):
        rcpsp_h = RCPSP_H_Model(base_rcpsp_model=the_model,
                                pre_helper_activities=pre_helper_activities,
                                post_helper_activities=post_helper_activities)
    elif isinstance(the_model, MultiModeRCPSPModel):  # Need to check the multi-mode works
        rcpsp_h = MRCPSP_H_Model(base_rcpsp_model=the_model,
                                 pre_helper_activities=pre_helper_activities,
                                 post_helper_activities=post_helper_activities)
    graph_precedence = rcpsp_h.compute_graph()
    predecessors_sink = graph_precedence.precedessors_nodes(new_sink_activity_id)
    if len(predecessors_sink) != rcpsp_h.n_jobs+1:
        print("Missing precedences to sink ")
        missing = [i for i in range(1, rcpsp_h.n_jobs+2) if i not in predecessors_sink]
        for m in missing:
            rcpsp_h.successors[m].append(new_sink_activity_id)  # this will insure the sink is the last activity...
    return rcpsp_h
