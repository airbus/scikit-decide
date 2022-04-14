import random
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from skdecide import DiscreteDistribution, Distribution, rollout_episode
from skdecide.builders.domain.scheduling.modes import (
    ConstantModeConsumption,
    ModeConsumption,
)
from skdecide.builders.domain.scheduling.scheduling_domains import (
    MultiModeRCPSPWithCost,
    SchedulingObjectiveEnum,
    SingleModeRCPSP,
    SingleModeRCPSP_Simulated_Stochastic_Durations_WithConditionalTasks,
    SingleModeRCPSP_Stochastic_Durations,
    SingleModeRCPSP_Stochastic_Durations_WithConditionalTasks,
)


class MyExampleRCPSPDomain(SingleModeRCPSP):
    def _get_objectives(self) -> List[SchedulingObjectiveEnum]:
        return [SchedulingObjectiveEnum.MAKESPAN]

    def __init__(self):
        self.initialize_domain()

    def _get_max_horizon(self) -> int:
        return 50

    def _get_successors(self) -> Dict[int, List[int]]:
        return {1: [2, 3], 2: [4], 3: [5], 4: [5], 5: []}

    def _get_tasks_ids(self) -> Union[Set[int], Dict[int, Any], List[int]]:
        return {1, 2, 3, 4, 5}

    def _get_tasks_mode(self) -> Dict[int, ModeConsumption]:
        return {
            1: ConstantModeConsumption({"r1": 0, "r2": 0}),
            2: ConstantModeConsumption({"r1": 1, "r2": 1}),
            3: ConstantModeConsumption({"r1": 1, "r2": 0}),
            4: ConstantModeConsumption({"r1": 2, "r2": 1}),
            5: ConstantModeConsumption({"r1": 0, "r2": 0}),
        }

    def _get_resource_types_names(self) -> List[str]:
        return ["r1", "r2"]

    def _get_task_duration(
        self, task: int, mode: Optional[int] = 1, progress_from: Optional[float] = 0.0
    ) -> int:
        all_durations = {1: 0, 2: 5, 3: 6, 4: 4, 5: 0}
        return all_durations[task]

    def _get_original_quantity_resource(self, resource: str, **kwargs) -> int:
        all_resource_quantities = {"r1": 2, "r2": 1}
        return all_resource_quantities[resource]


class MyExampleMRCPSPDomain_WithCost(MultiModeRCPSPWithCost):
    def _get_resource_renewability(self) -> Dict[str, bool]:
        return {"r1": True, "r2": True}

    def _get_tasks_modes(self) -> Dict[int, Dict[int, ModeConsumption]]:
        return {
            1: {1: ConstantModeConsumption({"r1": 0, "r2": 0})},
            2: {
                1: ConstantModeConsumption({"r1": 1, "r2": 1}),
                2: ConstantModeConsumption({"r1": 2, "r2": 0}),
            },
            3: {
                1: ConstantModeConsumption({"r1": 1, "r2": 0}),
                2: ConstantModeConsumption({"r1": 0, "r2": 1}),
            },
            4: {
                1: ConstantModeConsumption({"r1": 2, "r2": 1}),
                2: ConstantModeConsumption({"r1": 2, "r2": 0}),
            },
            5: {1: ConstantModeConsumption({"r1": 0, "r2": 0})},
        }

    def _get_mode_costs(self) -> Dict[int, Dict[int, float]]:
        return {
            1: {1: 0.0},
            2: {1: 1.0, 2: 2.0},
            3: {1: 1.0, 2: 1.0},
            4: {1: 0.0, 2: 1.0},
            5: {1: 0.0},
        }

    def _get_resource_cost_per_time_unit(self) -> Dict[str, float]:
        return {"r1": 1, "r2": 2}

    def _get_objectives(self) -> List[SchedulingObjectiveEnum]:
        return [SchedulingObjectiveEnum.COST]

    def __init__(self):
        self.initialize_domain()

    def _get_max_horizon(self) -> int:
        return 50

    def _get_successors(self) -> Dict[int, List[int]]:
        return {1: [2, 3], 2: [4], 3: [5], 4: [5], 5: []}

    def _get_tasks_ids(self) -> Union[Set[int], Dict[int, Any], List[int]]:
        return set([1, 2, 3, 4, 5])

    def _get_resource_types_names(self) -> List[str]:
        return ["r1", "r2"]

    def _get_task_duration(
        self, task: int, mode: Optional[int] = 1, progress_from: Optional[float] = 0.0
    ) -> int:
        all_durations = {1: 0, 2: 5, 3: 6, 4: 4, 5: 0}
        return all_durations[task]

    def _get_original_quantity_resource(self, resource: str, **kwargs) -> int:
        all_resource_quantities = {"r1": 2, "r2": 1}
        return all_resource_quantities[resource]


class MyExampleSRCPSPDomain(SingleModeRCPSP_Stochastic_Durations):
    def _get_task_duration_distribution(
        self,
        task: int,
        mode: Optional[int] = 1,
        progress_from: Optional[float] = 0.0,
        multivariate_settings: Optional[Dict[str, int]] = None,
    ) -> Distribution:
        all_distributions = {}
        all_distributions[1] = DiscreteDistribution([(0, 1.0)])
        all_distributions[2] = DiscreteDistribution([(4, 0.25), (5, 0.5), (6, 0.25)])
        all_distributions[3] = DiscreteDistribution([(5, 0.25), (6, 0.5), (7, 0.25)])
        all_distributions[4] = DiscreteDistribution([(3, 0.5), (4, 0.5)])
        all_distributions[5] = DiscreteDistribution([(0, 1.0)])
        return all_distributions[task]

    def _get_objectives(self) -> List[SchedulingObjectiveEnum]:
        return [SchedulingObjectiveEnum.MAKESPAN]

    def __init__(self):
        self.initialize_domain()

    def _get_max_horizon(self) -> int:
        return 20

    def _get_successors(self) -> Dict[int, List[int]]:
        return {1: [2, 3], 2: [4], 3: [5], 4: [5], 5: []}

    def _get_tasks_ids(self) -> Union[Set[int], Dict[int, Any], List[int]]:
        return set([1, 2, 3, 4, 5])

    def _get_tasks_mode(self) -> Dict[int, ModeConsumption]:
        return {
            1: ConstantModeConsumption({"r1": 0, "r2": 0}),
            2: ConstantModeConsumption({"r1": 1, "r2": 1}),
            3: ConstantModeConsumption({"r1": 1, "r2": 0}),
            4: ConstantModeConsumption({"r1": 2, "r2": 1}),
            5: ConstantModeConsumption({"r1": 0, "r2": 0}),
        }

    def _get_resource_types_names(self) -> List[str]:
        return ["r1", "r2"]

    def _get_original_quantity_resource(self, resource: str, **kwargs) -> int:
        all_resource_quantities = {"r1": 2, "r2": 1}
        return all_resource_quantities[resource]


class MyExampleSRCPSPDomain_2(SingleModeRCPSP_Stochastic_Durations):
    def _get_task_duration_distribution(
        self,
        task: int,
        mode: Optional[int] = 1,
        progress_from: Optional[float] = 0.0,
        multivariate_settings: Optional[Dict[str, int]] = None,
    ) -> Distribution:
        all_distributions = {}
        t = None
        if multivariate_settings is not None:
            if "t" in multivariate_settings:
                t = multivariate_settings["t"]
        all_distributions[1] = DiscreteDistribution([(0, 1.0)])
        all_distributions[2] = DiscreteDistribution([(4, 0.25), (5, 0.5), (6, 0.25)])
        if t is not None:
            if t == 1:
                all_distributions[2] = DiscreteDistribution(
                    [(0, 0.25), (1, 0.75)]
                )  # Faster
        all_distributions[3] = DiscreteDistribution([(5, 0.25), (6, 0.5), (7, 0.25)])
        all_distributions[4] = DiscreteDistribution([(3, 0.5), (4, 0.5)])
        all_distributions[5] = DiscreteDistribution([(0, 1.0)])
        return all_distributions[task]

    def _get_objectives(self) -> List[SchedulingObjectiveEnum]:
        return [SchedulingObjectiveEnum.MAKESPAN]

    def __init__(self):
        self.initialize_domain()

    def _get_max_horizon(self) -> int:
        return 20

    def _get_successors(self) -> Dict[int, List[int]]:
        return {1: [2, 3], 2: [4], 3: [5], 4: [5], 5: []}

    def _get_tasks_ids(self) -> Union[Set[int], Dict[int, Any], List[int]]:
        return set([1, 2, 3, 4, 5])

    def _get_tasks_mode(self) -> Dict[int, ModeConsumption]:
        return {
            1: ConstantModeConsumption({"r1": 0, "r2": 0}),
            2: ConstantModeConsumption({"r1": 1, "r2": 1}),
            3: ConstantModeConsumption({"r1": 1, "r2": 0}),
            4: ConstantModeConsumption({"r1": 2, "r2": 1}),
            5: ConstantModeConsumption({"r1": 0, "r2": 0}),
        }

    def _get_resource_types_names(self) -> List[str]:
        return ["r1", "r2"]

    def _get_original_quantity_resource(self, resource: str, **kwargs) -> int:
        all_resource_quantities = {"r1": 2, "r2": 1}
        return all_resource_quantities[resource]


class ConditionElementsExample(Enum):
    OK = 0
    PROBLEM_OPERATION_2 = 1
    PROBLEM_OPERATION_3 = 2


class MyExampleCondSRCPSPDomain(
    SingleModeRCPSP_Stochastic_Durations_WithConditionalTasks
):
    def _get_all_condition_items(self) -> Enum:
        return ConditionElementsExample

    def _get_task_on_completion_added_conditions(self) -> Dict[int, List[Distribution]]:
        completion_conditions_dict = {}

        completion_conditions_dict[2] = [
            DiscreteDistribution(
                [
                    (ConditionElementsExample.PROBLEM_OPERATION_2, 0.1),
                    (ConditionElementsExample.OK, 0.9),
                ]
            )
        ]
        completion_conditions_dict[3] = [
            DiscreteDistribution(
                [
                    (ConditionElementsExample.PROBLEM_OPERATION_3, 0.9),
                    (ConditionElementsExample.OK, 0.1),
                ]
            )
        ]

        # completion_conditions_dict[2] = [
        #     DiscreteDistribution(
        #         [(ConditionElementsExample.PROBLEM_OPERATION_2, 1)])
        # ]
        # completion_conditions_dict[3] = [
        #     DiscreteDistribution(
        #         [(ConditionElementsExample.PROBLEM_OPERATION_3, 1)])
        # ]

        return completion_conditions_dict

    def _get_task_existence_conditions(self) -> Dict[int, List[int]]:
        existence_conditions_dict = {
            5: [self.get_all_condition_items().PROBLEM_OPERATION_2],
            6: [self.get_all_condition_items().PROBLEM_OPERATION_3],
        }
        return existence_conditions_dict

    def _get_task_duration_distribution(
        self,
        task: int,
        mode: Optional[int] = 1,
        progress_from: Optional[float] = 0.0,
        multivariate_settings: Optional[Dict[str, int]] = None,
    ) -> Distribution:
        all_distributions = {}
        all_distributions[1] = DiscreteDistribution([(0, 1.0)])
        all_distributions[2] = DiscreteDistribution([(4, 0.25), (5, 0.5), (6, 0.25)])
        all_distributions[3] = DiscreteDistribution([(5, 0.25), (6, 0.5), (7, 0.25)])
        all_distributions[4] = DiscreteDistribution([(3, 0.5), (4, 0.5)])
        all_distributions[5] = DiscreteDistribution([(4, 0.5), (5, 0.5)])
        all_distributions[6] = DiscreteDistribution([(2, 0.5), (3, 0.5)])
        all_distributions[7] = DiscreteDistribution([(0, 1.0)])

        return all_distributions[task]

    def _get_objectives(self) -> List[SchedulingObjectiveEnum]:
        return [SchedulingObjectiveEnum.MAKESPAN]

    def __init__(self):
        self.initialize_domain()

    def _get_max_horizon(self) -> int:
        return 50

    def _get_successors(self) -> Dict[int, List[int]]:
        return {1: [2, 3], 2: [4], 3: [7], 4: [7], 5: [7], 6: [7], 7: []}

    def _get_tasks_ids(self) -> Union[Set[int], Dict[int, Any], List[int]]:
        return {1, 2, 3, 4, 5, 6, 7}

    def _get_tasks_mode(self) -> Dict[int, ModeConsumption]:
        return {
            1: ConstantModeConsumption({"r1": 0, "r2": 0}),
            2: ConstantModeConsumption({"r1": 1, "r2": 1}),
            3: ConstantModeConsumption({"r1": 1, "r2": 0}),
            4: ConstantModeConsumption({"r1": 2, "r2": 1}),
            5: ConstantModeConsumption({"r1": 0, "r2": 1}),
            6: ConstantModeConsumption({"r1": 1, "r2": 0}),
            7: ConstantModeConsumption({"r1": 0, "r2": 0}),
        }

    def _get_resource_types_names(self) -> List[str]:
        return ["r1", "r2"]

    def _get_original_quantity_resource(self, resource: str, **kwargs) -> int:
        all_resource_quantities = {"r1": 2, "r2": 1}
        return all_resource_quantities[resource]


class MyExampleSimulatedCondSRCPSPDomain(
    SingleModeRCPSP_Simulated_Stochastic_Durations_WithConditionalTasks
):
    def _get_all_condition_items(self) -> Enum:
        return ConditionElementsExample

    def _get_task_on_completion_added_conditions(self) -> Dict[int, List[Distribution]]:
        completion_conditions_dict = {}

        completion_conditions_dict[2] = [
            DiscreteDistribution(
                [
                    (ConditionElementsExample.PROBLEM_OPERATION_2, 0.1),
                    (ConditionElementsExample.OK, 0.9),
                ]
            )
        ]
        completion_conditions_dict[3] = [
            DiscreteDistribution(
                [
                    (ConditionElementsExample.PROBLEM_OPERATION_3, 0.9),
                    (ConditionElementsExample.OK, 0.1),
                ]
            )
        ]

        # completion_conditions_dict[2] = [
        #     DiscreteDistribution(
        #         [(ConditionElementsExample.PROBLEM_OPERATION_2, 1)])
        # ]
        # completion_conditions_dict[3] = [
        #     DiscreteDistribution(
        #         [(ConditionElementsExample.PROBLEM_OPERATION_3, 1)])
        # ]

        return completion_conditions_dict

    def _get_task_existence_conditions(self) -> Dict[int, List[int]]:
        existence_conditions_dict = {
            5: [self.get_all_condition_items().PROBLEM_OPERATION_2],
            6: [self.get_all_condition_items().PROBLEM_OPERATION_3],
        }
        return existence_conditions_dict

    def _get_objectives(self) -> List[SchedulingObjectiveEnum]:
        return [SchedulingObjectiveEnum.MAKESPAN]

    def __init__(self):
        self.initialize_domain()

    def _get_max_horizon(self) -> int:
        return 50

    def _get_successors(self) -> Dict[int, List[int]]:
        return {1: [2, 3], 2: [4], 3: [7], 4: [7], 5: [7], 6: [7], 7: []}

    def _get_tasks_ids(self) -> Union[Set[int], Dict[int, Any], List[int]]:
        return set([1, 2, 3, 4, 5, 6, 7])

    def _get_tasks_mode(self) -> Dict[int, ModeConsumption]:
        return {
            1: ConstantModeConsumption({"r1": 0, "r2": 0}),
            2: ConstantModeConsumption({"r1": 1, "r2": 1}),
            3: ConstantModeConsumption({"r1": 1, "r2": 0}),
            4: ConstantModeConsumption({"r1": 2, "r2": 1}),
            5: ConstantModeConsumption({"r1": 0, "r2": 1}),
            6: ConstantModeConsumption({"r1": 1, "r2": 0}),
            7: ConstantModeConsumption({"r1": 0, "r2": 0}),
        }

    def _get_resource_types_names(self) -> List[str]:
        return ["r1", "r2"]

    def _get_original_quantity_resource(self, resource: str, **kwargs) -> int:
        all_resource_quantities = {"r1": 2, "r2": 1}
        return all_resource_quantities[resource]

    def _sample_task_duration(
        self, task: int, mode: Optional[int] = 1, progress_from: Optional[float] = 0.0
    ) -> int:
        val = random.randint(3, 6)
        return val


def run_example():
    # domain = MyExampleRCPSPDomain()
    # domain = MyExampleMRCPSPDomain_WithCost()
    # domain = MyExampleSRCPSPDomain()
    # domain = MyExampleCondSRCPSPDomain()
    domain = MyExampleSimulatedCondSRCPSPDomain()

    state = domain.get_initial_state()
    print("Initial state : ", state)
    # actions = domain.get_applicable_actions(state)
    # print([str(action) for action in actions.get_elements()])
    # action = actions.get_elements()[0]
    # new_state = domain.get_next_state(state, action)
    # print("New state ", new_state)
    # actions = domain.get_applicable_actions(new_state)
    # print("New actions : ", [str(action) for action in actions.get_elements()])
    # action = actions.get_elements()[0]
    # print(action)
    # new_state = domain.get_next_state(new_state, action)
    # print("New state :", new_state)
    # print('_is_terminal: ', domain._is_terminal(state))
    # ONLY KEEP LINE BELOW FOR SIMPLE ROLLOUT
    solver = None
    # UNCOMMENT BELOW TO USE ASTAR
    # domain.set_inplace_environment(False)
    # solver = lazy_astar.LazyAstar(from_state=state, heuristic=None, verbose=True)
    # solver.solve(domain_factory=lambda: domain)
    states, actions, values = rollout_episode(
        domain=domain,
        max_steps=1000,
        solver=solver,
        from_memory=state,
        outcome_formatter=lambda o: f"{o.observation} - cost: {o.value.cost:.2f}",
    )
    print(states[-1])


def run_astar():
    from skdecide.hub.solver.lazy_astar import LazyAstar

    domain = MyExampleRCPSPDomain()
    # domain = MyExampleSRCPSPDomain()
    domain.set_inplace_environment(False)
    state = domain.get_initial_state()
    print("Initial state : ", state)
    solver = LazyAstar(from_state=state, heuristic=None, verbose=True)
    solver.solve(domain_factory=lambda: domain)
    states, actions, values = rollout_episode(
        domain=domain,
        max_steps=1000,
        solver=solver,
        from_memory=state,
        outcome_formatter=lambda o: f"{o.observation} - cost: {o.value.cost:.2f}",
    )
    print("Cost :", sum([v.cost for v in values]))

    from skdecide.hub.solver.do_solver.sk_to_do_binding import (
        from_last_state_to_solution,
    )

    do_sol = from_last_state_to_solution(states[-1], domain)
    from discrete_optimization.rcpsp.rcpsp_plot_utils import (
        plot_resource_individual_gantt,
        plot_ressource_view,
        plot_task_gantt,
        plt,
    )

    plot_task_gantt(do_sol.problem, do_sol)
    plot_ressource_view(do_sol.problem, do_sol)
    plot_resource_individual_gantt(do_sol.problem, do_sol)
    plt.show()


def run_do():
    from skdecide.hub.solver.do_solver.do_solver_scheduling import (
        BasePolicyMethod,
        DOSolver,
        PolicyMethodParams,
        SolvingMethod,
    )

    domain = MyExampleRCPSPDomain()

    domain.set_inplace_environment(False)
    state = domain.get_initial_state()
    print("Initial state : ", state)
    solver = DOSolver(
        policy_method_params=PolicyMethodParams(
            base_policy_method=BasePolicyMethod.SGS_PRECEDENCE,
            delta_index_freedom=0,
            delta_time_freedom=0,
        ),
        method=SolvingMethod.LNS_CP_CALENDAR,
    )
    solver.solve(domain_factory=lambda: domain)
    states, actions, values = rollout_episode(
        domain=domain,
        max_steps=1000,
        solver=solver,
        from_memory=state,
        action_formatter=lambda o: str(o),
        outcome_formatter=lambda o: f"{o.observation} - cost: {o.value.cost:.2f}",
    )
    print("Cost :", sum([v.cost for v in values]))
    from skdecide.hub.solver.do_solver.sk_to_do_binding import (
        from_last_state_to_solution,
    )

    do_sol = from_last_state_to_solution(states[-1], domain)
    from discrete_optimization.rcpsp.rcpsp_plot_utils import (
        plot_resource_individual_gantt,
        plot_ressource_view,
        plot_task_gantt,
        plt,
    )

    plot_task_gantt(do_sol.problem, do_sol)
    plot_ressource_view(do_sol.problem, do_sol)
    plot_resource_individual_gantt(do_sol.problem, do_sol)
    plt.show()


def run_graph_exploration():
    domain = MyExampleSRCPSPDomain_2()
    domain.set_inplace_environment(False)
    state = domain.get_initial_state()
    print("Initial state : ", state)
    from skdecide.hub.solver.graph_explorer.DFS_Uncertain_Exploration import (
        DFSExploration,
    )

    explorer = DFSExploration(domain=domain, max_edges=10000, max_nodes=10000)
    graph_exploration = explorer.build_graph_domain(init_state=state)
    nx_graph = graph_exploration.to_networkx()
    for state in graph_exploration.next_state_map:
        for action in graph_exploration.next_state_map[state]:
            new_state = list(graph_exploration.next_state_map[state][action].keys())
            if len(new_state) > 1:
                print(len(new_state))
                print(action, " leading to ", len(new_state), " new states ")
                print(
                    [
                        graph_exploration.next_state_map[state][action][s]
                        for s in new_state
                    ]
                )
    print("graph done")
    print(nx_graph.number_of_nodes())
    print(nx_graph.size())

    goal_states = [
        s for s in graph_exploration.state_goal if graph_exploration.state_goal[s]
    ]
    for task in goal_states[0].tasks_complete:
        durations = [
            s.tasks_details[task].end - s.tasks_details[task].start for s in goal_states
        ]
        print("Duration of task ", task, set(durations))
    for goal_state in goal_states:
        print(goal_state.t, " as makespan")


def run_graph_exploration_conditional():
    domain = MyExampleCondSRCPSPDomain()
    domain.set_inplace_environment(False)
    state = domain.get_initial_state()
    print("Initial state : ", state)
    from itertools import count

    from skdecide.hub.solver.graph_explorer.DFS_Uncertain_Exploration import (
        DFSExploration,
    )

    c = count()
    score_state = lambda x: (
        len(x.tasks_remaining) + len(x.tasks_ongoing) + len(x.tasks_complete),
        len(x.tasks_remaining),
        -len(x.tasks_complete),
        -len(x.tasks_ongoing),
        x.t,
        next(c),
    )
    # score_state = lambda x: (len(x.tasks_remaining),
    #                          -len(x.tasks_complete),
    #                          -len(x.tasks_ongoing),
    #                          x.t,
    #                          next(c))
    explorer = DFSExploration(
        domain=domain, max_edges=30000, score_function=score_state, max_nodes=30000
    )
    graph_exploration = explorer.build_graph_domain(init_state=state)
    nx_graph = graph_exploration.to_networkx()
    for state in graph_exploration.next_state_map:
        for action in graph_exploration.next_state_map[state]:
            new_state = list(graph_exploration.next_state_map[state][action].keys())
            if len(new_state) > 1:
                print(len(new_state))
                print(action, " leading to ", len(new_state), " new states ")
                print(
                    [
                        graph_exploration.next_state_map[state][action][s]
                        for s in new_state
                    ]
                )
    print("graph done")
    print(nx_graph.number_of_nodes())
    print(nx_graph.size())

    for g in graph_exploration.state_goal:
        print(g.tasks_remaining, g.tasks_complete)
        print(7 in g.tasks_complete)
        if 7 in g.tasks_complete:
            print("should be final")
            print(domain.is_goal(g))
        print(graph_exploration.state_goal[g])
    goal_states = [
        s for s in graph_exploration.state_goal if graph_exploration.state_goal[s]
    ]
    for task in goal_states[0].task_ids:
        durations = [
            s.tasks_details[task].end - s.tasks_details[task].start
            if s.tasks_details[task].end is not None
            else None
            for s in goal_states
        ]
        print("Duration of task ", task, set(durations))
    for goal_state in goal_states:
        print(goal_state.t, " as makespan")
        print(len(goal_state.tasks_complete))
    print(set([len(goal_state.tasks_complete) for goal_state in goal_states]))


if __name__ == "__main__":
    run_do()
    run_astar()
    # run_example()
    # run_graph_exploration_conditional()
