from skdecide.builders.domain.scheduling.scheduling_domains_modelling import SchedulingAction, SchedulingActionEnum, State

from skdecide import rollout_episode, DiscreteDistribution

from skdecide.hub.domain.rcpsp.rcpsp_sk_parser import load_domain, load_multiskill_domain
from skdecide.hub.domain.rcpsp.rcpsp_sk import RCPSP, MSRCPSP
from skdecide.hub.solver.do_solver.do_solver_scheduling import DOSolver, SolvingMethod
from skdecide.hub.solver.graph_explorer.DFS_Uncertain_Exploration import DFSExploration
from skdecide.hub.solver.sgs_policies.sgs_policies import PolicyMethodParams, BasePolicyMethod
from tests.test_scheduling import ToyMS_RCPSPDomain, ToySRCPSPDomain
from examples.discrete_optimization.rcpsp_parser_example import get_complete_path, get_data_available
from examples.discrete_optimization.rcpsp_multiskill_parser_example import get_data_available_ms, \
    get_complete_path_ms


def random_walk():
    domain: RCPSP = load_domain(get_complete_path("j301_1.sm"))
    state = domain.get_initial_state()
    domain.set_inplace_environment(False)
    states, actions, values = rollout_episode(domain=domain,
                                              solver=None,
                                              from_memory=state,
                                              max_steps=500,
                                              outcome_formatter=lambda o: f'{o.observation} - cost: {o.value.cost:.2f}')
    print(sum([v.cost for v in values]))
    print("rollout done")
    print('end times: ')
    for task_id in states[-1].tasks_details.keys():
        print('end task', task_id, ': ', states[-1].tasks_details[task_id].end)
    from skdecide.discrete_optimization.rcpsp.rcpsp_plot_utils import plot_resource_individual_gantt, plt
    from skdecide.discrete_optimization.rcpsp.rcpsp_plot_utils import plot_task_gantt
    from skdecide.discrete_optimization.rcpsp.rcpsp_plot_utils import plot_ressource_view
    from skdecide.hub.solver.do_solver.sk_to_do_binding import from_last_state_to_solution
    do_sol = from_last_state_to_solution(states[-1], domain)
    plot_task_gantt(do_sol.problem, do_sol)
    plot_ressource_view(do_sol.problem, do_sol)
    plot_resource_individual_gantt(do_sol.problem, do_sol)
    plt.show()


def cp_solve():
    do_solver = SolvingMethod.CP
    domain: RCPSP = load_domain(get_complete_path("j301_1.sm"))
    domain.set_inplace_environment(False)
    state = domain.get_initial_state()
    print("Initial state : ", state)
    solver = DOSolver(policy_method_params=PolicyMethodParams(base_policy_method=BasePolicyMethod.SGS_PRECEDENCE,
                                                              delta_index_freedom=0,
                                                              delta_time_freedom=0),
                      method=do_solver)
    solver.solve(domain_factory=lambda: domain)
    print(do_solver)

    states, actions, values = rollout_episode(domain=domain,
                                              solver=solver,
                                              from_memory=state,
                                              max_steps=500,
                                              outcome_formatter=lambda o: f'{o.observation} - cost: {o.value.cost:.2f}')
    print(sum([v.cost for v in values]))
    print("rollout done")
    print('end times: ')
    for task_id in states[-1].tasks_details.keys():
        print('end task', task_id, ': ', states[-1].tasks_details[task_id].end)


def do_multimode():
    domain: RCPSP = load_domain(get_complete_path("j1010_2.mm"))
    state = domain.get_initial_state()
    solver = DOSolver(policy_method_params=PolicyMethodParams(base_policy_method=BasePolicyMethod.FOLLOW_GANTT,
                                                              delta_index_freedom=0,
                                                              delta_time_freedom=0),
                      method=SolvingMethod.CP)
    solver.solve(domain_factory=lambda: domain)
    states, actions, values = rollout_episode(domain=domain,
                                              solver=solver,
                                              from_memory=state,
                                              max_steps=1000,
                                              action_formatter=lambda a: f'{a}',
                                              outcome_formatter=lambda o: f'{o.observation} - cost: {o.value.cost:.2f}')
    print("rollout done")
    print('end times: ')
    for task_id in states[-1].tasks_details.keys():
        print('end task', task_id, ': ', states[-1].tasks_details[task_id].end)


def random_walk_multiskill():
    domain: MSRCPSP = load_multiskill_domain(get_data_available_ms()[0])
    state = domain.get_initial_state()
    states, actions, values = rollout_episode(domain=domain,
                                              solver=None,
                                              from_memory=state,
                                              max_steps=1000,
                                              action_formatter=lambda a: f'{a}',
                                              outcome_formatter=lambda o: f'{o.observation} - cost: {o.value.cost:.2f}')
    print("rollout done")
    print('end times: ')
    for task_id in states[-1].tasks_details.keys():
        print('end task', task_id, ': ', states[-1].tasks_details[task_id].end)


def do_multiskill():
    domain: MSRCPSP = load_multiskill_domain(get_data_available_ms()[0])
    domain.set_inplace_environment(False)
    state = domain.get_initial_state()
    solver = DOSolver(policy_method_params=PolicyMethodParams(base_policy_method=BasePolicyMethod.SGS_PRECEDENCE,
                                                              delta_index_freedom=0,
                                                              delta_time_freedom=0),
                      method=SolvingMethod.LNS_CP)
    solver.get_available_methods(domain)
    solver.solve(domain_factory=lambda: domain)
    states, actions, values = rollout_episode(domain=domain,
                                              solver=solver,
                                              from_memory=state,
                                              max_steps=3000,
                                              action_formatter=lambda a: f'{a}',
                                              outcome_formatter=lambda o: f'{o.observation} - cost: {o.value.cost:.2f}')
    print(sum([v.cost for v in values]))
    print("rollout done")
    print('end times: ')
    for task_id in states[-1].tasks_details.keys():
        print('end task', task_id, ': ', states[-1].tasks_details[task_id].end)


def do_multiskill_toy():
    do_domain = ToyMS_RCPSPDomain()
    solver = DOSolver(policy_method_params=PolicyMethodParams(base_policy_method=BasePolicyMethod.SGS_PRECEDENCE,
                                                              delta_index_freedom=0,
                                                              delta_time_freedom=0),
                      method=SolvingMethod.LNS_CP)
    solver.solve(domain_factory=lambda: do_domain)


def check_uncertain_domain():
    from itertools import count
    domain = ToySRCPSPDomain()
    c = count()
    score_state = lambda x: (len(x.tasks_remaining)
                             + len(x.tasks_ongoing)
                             + len(x.tasks_complete),
                             len(x.tasks_remaining),
                             -len(x.tasks_complete),
                             -len(x.tasks_ongoing),
                             x.t,
                             next(c))
    domain.set_inplace_environment(False)
    state = domain.get_initial_state()
    explorer = DFSExploration(domain=domain,
                              max_edges=30000,
                              score_function=score_state,
                              max_nodes=30000)
    graph_exploration = explorer.build_graph_domain(init_state=state)
    for state in graph_exploration.next_state_map:
        for action in graph_exploration.next_state_map[state]:
            ac: SchedulingAction = action
            if ac.action == SchedulingActionEnum.START:
                task = ac.task
                duration_distribution: DiscreteDistribution = domain.get_task_duration_distribution(task)
                values = duration_distribution.get_values()
                new_states = list(graph_exploration.next_state_map[state][action].keys())
                task_duration = {v[0]: v[1] for v in values}
                assert len(new_states) == len(task_duration)  # as many states as possible task duration
                for ns in new_states:
                    ns: State = ns
                    prob, cost = graph_exploration.next_state_map[state][action][ns]
                    duration_task_for_ns = ns.tasks_details[task].sampled_duration
                    print(prob, duration_task_for_ns)
                    assert duration_task_for_ns in task_duration and \
                           prob == task_duration[
                               duration_task_for_ns]  # duration are coherent with the input distribution


def run_and_compare_policies():
    domain: RCPSP = load_domain(get_complete_path("j1010_2.mm"))
    state = domain.get_initial_state()
    solver = DOSolver(policy_method_params=PolicyMethodParams(base_policy_method=BasePolicyMethod.FOLLOW_GANTT,
                                                              delta_index_freedom=0,
                                                              delta_time_freedom=0),
                      method=SolvingMethod.CP)
    solver.solve(domain_factory=lambda: domain)
    policy_methods = [PolicyMethodParams(base_policy_method=method,
                                         delta_time_freedom=0,
                                         delta_index_freedom=0)
                      for method in [BasePolicyMethod.FOLLOW_GANTT,
                                     BasePolicyMethod.SGS_PRECEDENCE, #,
                                     BasePolicyMethod.SGS_READY
                                     #BasePolicyMethod.SGS_STRICT
                                     ]
                      ]
    # policy_methods += [PolicyMethodParams(base_policy_method=BasePolicyMethod.SGS_INDEX_FREEDOM,
    #                                       delta_time_freedom=0,
    #                                       delta_index_freedom=i)
    #                    for i in range(10)]
    # policy_methods += [PolicyMethodParams(base_policy_method=BasePolicyMethod.SGS_TIME_FREEDOM,
    #                                       delta_time_freedom=t,
    #                                       delta_index_freedom=0)
    #                    for t in range(0, 200, 5)]

    policies = {i: solver.compute_external_policy(policy_methods[i])
                for i in range(len(policy_methods))}
    from skdecide.hub.solver.meta_policy.meta_policies import MetaPolicy
    meta_policy = MetaPolicy(policies=policies,
                             execution_domain=domain,
                             known_domain=domain,
                             nb_rollout_estimation=1,
                             verbose=True)
    states, actions, values = rollout_episode(domain=domain,
                                              solver=meta_policy,
                                              from_memory=state,
                                              max_steps=1000,
                                              action_formatter=lambda a: f'{a}',
                                              outcome_formatter=lambda o: f'{o.observation} - cost: {o.value.cost:.2f}')


if __name__ == "__main__":
    # check_uncertain_domain()
    #random_walk()
    #cp_solve()
    #random_walk_multiskill()
    do_multiskill_toy()
    #do_multimode()
    # random_walk()
    # do_multiskill_toy()
    # cp_solve()
