from discrete_optimization.generic_tasks_tools.generic_scheduling_impl import (
    GenericSchedulingImplProblem,
)

from skdecide.hub.domain.scheduling.do_to_sk_binding import build_sk_domain


def test_as_do_generic_scheduling():
    problem = GenericSchedulingImplProblem(
        horizon=20,
        durations_per_mode={
            "task-1": {
                0: 1,
                1: 3,
            },
            "task-2": {
                0: 4,
            },
            "task-3": {
                0: 2,
                1: 2,
            },
        },
        resource_consumptions={
            "task-1": {
                0: {
                    "non_renewable_resource": 2,
                },
                1: {
                    "non_renewable_resource": 1,
                },
            },
            "task-2": {
                0: {
                    "cumulative_resource": 2,
                },
            },
            "task-3": {
                0: {
                    "cumulative_resource": 1,
                },
                1: {
                    "non_renewable_resource": 1,
                },
            },
        },
        successors={"task-1": ["task-2", "task-3"]},
        end_to_start_min_time_lags=[("task-3", "task-2", -5)],
        unary_resources={"worker1", "worker2"},
        unary_resources_availabilities={
            "worker1": [(1, 4)],
            "worker2": [(3, 18)],
        },
        non_skill_cumulative_resources={
            "cumulative_resource": [
                (3, 5, 1),
                (5, 12, 2),
                (12, 20, 3),
            ],
        },
        non_renewable_resources={
            "non_renewable_resource": 1,
        },
    )
    domain = build_sk_domain(problem)
    state = domain.get_initial_state()
    actions = domain.get_applicable_actions(state)
    action = actions.sample()
    new_state = domain.get_next_state(state, action)
    actions = domain.get_applicable_actions(new_state)
    action = actions.sample()
    new_state = domain.get_next_state(new_state, action)
    domain.is_terminal(new_state)
