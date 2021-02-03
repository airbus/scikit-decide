import pytest

from enum import Enum
from typing import Dict, List, Any, Optional, Union, Set
import random

from skdecide import Distribution
from skdecide.builders.scheduling.scheduling_domains import SingleModeRCPSP, SchedulingDomain, \
    SchedulingObjectiveEnum, SingleModeRCPSP_Stochastic_Durations, \
    SingleModeRCPSP_Stochastic_Durations_WithConditionalTasks, \
    SingleModeRCPSP_Simulated_Stochastic_Durations_WithConditionalTasks, \
    MultiModeRCPSPWithCost, MultiModeMultiSkillRCPSPCalendar, MultiModeMultiSkillRCPSP, State, SchedulingAction
from skdecide import Domain, Space, TransitionValue, Distribution, TransitionOutcome, ImplicitSpace, DiscreteDistribution

from skdecide.builders.scheduling.modes import SingleMode, MultiMode, ModeConsumption, ConstantModeConsumption
from skdecide.builders.scheduling.resource_consumption import VariableResourceConsumption, ConstantResourceConsumption
from skdecide.builders.scheduling.precedence import WithPrecedence, WithoutPrecedence
from skdecide.builders.scheduling.preemptivity import WithPreemptivity, WithoutPreemptivity, ResumeType
from skdecide.builders.scheduling.resource_type import WithResourceTypes, WithoutResourceTypes, WithResourceUnits, WithoutResourceUnit, SingleResourceUnit
from skdecide.builders.scheduling.resource_renewability import RenewableOnly, MixedRenewable
from skdecide.builders.scheduling.scheduling_domains_modelling import SchedulingActionEnum
from skdecide.builders.scheduling.task_duration import SimulatedTaskDuration, DeterministicTaskDuration, UncertainUnivariateTaskDuration
from skdecide.builders.scheduling.task_progress import CustomTaskProgress, DeterministicTaskProgress
from skdecide.builders.scheduling.skills import WithResourceSkills, WithoutResourceSkills
from skdecide.builders.scheduling.time_lag import WithTimeLag, WithoutTimeLag, TimeLag
from skdecide.builders.scheduling.time_windows import WithTimeWindow, WithoutTimeWindow, TimeWindow
from skdecide.builders.scheduling.preallocations import WithPreallocations, WithoutPreallocations
from skdecide.builders.scheduling.conditional_tasks import WithConditionalTasks, WithoutConditionalTasks
from skdecide.builders.scheduling.resource_availability import UncertainResourceAvailabilityChanges, DeterministicResourceAvailabilityChanges, WithoutResourceAvailabilityChange
from skdecide import rollout, rollout_episode
from skdecide.hub.domain.rcpsp.rcpsp_sk import RCPSP, MRCPSP, build_n_determinist_from_stochastic
from skdecide.hub.domain.rcpsp.rcpsp_sk_parser import load_domain, load_multiskill_domain
from skdecide.hub.solver.graph_explorer.DFS_Uncertain_Exploration import DFSExploration
from skdecide.hub.solver.lazy_astar import lazy_astar
from skdecide.hub.solver.do_solver.do_solver_scheduling import PolicyRCPSP, DOSolver, \
    PolicyMethodParams, BasePolicyMethod, SolvingMethod
from skdecide.hub.solver.lazy_astar import LazyAstar
from skdecide.hub.solver.gphh.gphh import GPHH, ParametersGPHH

optimal_solutions = {
    'ToyRCPSPDomain': {'makespan': 10},
    'ToyMS_RCPSPDomain': {'makespan': 10}
}


class ToyRCPSPDomain(SingleModeRCPSP):
    def _get_objectives(self) -> List[SchedulingObjectiveEnum]:
        return [SchedulingObjectiveEnum.MAKESPAN]

    def __init__(self):
        self.initialize_domain()

    def _get_max_horizon(self) -> int:
        return 50

    def _get_successors(self) -> Dict[int, List[int]]:
        return {1: [2,3], 2:[4], 3:[5], 4:[5], 5:[]}

    def _get_tasks_ids(self) -> Union[Set[int], Dict[int, Any], List[int]]:
        return set([1,2,3,4,5])

    def _get_tasks_mode(self) -> Dict[int, ModeConsumption]:
        return {
            1: ConstantModeConsumption({'r1': 0, 'r2': 0}),
            2: ConstantModeConsumption({'r1': 1, 'r2': 1}),
            3: ConstantModeConsumption({'r1': 1, 'r2': 0}),
            4: ConstantModeConsumption({'r1': 2, 'r2': 1}),
            5: ConstantModeConsumption({'r1': 0, 'r2': 0})
        }

    def _get_resource_types_names(self) -> List[str]:
        return ['r1', 'r2']

    def _get_task_duration(self, task: int, mode: Optional[int] = 1, progress_from: Optional[float] = 0.) -> int:
        all_durations = {1: 0, 2: 5, 3: 6, 4: 4, 5: 0}
        return all_durations[task]

    def _get_original_quantity_resource(self, resource: str, **kwargs) -> int:
        all_resource_quantities = {'r1': 2, 'r2': 1}
        return all_resource_quantities[resource]


class ToyMRCPSPDomain_WithCost(MultiModeRCPSPWithCost):

    def _get_resource_renewability(self) -> Dict[str, bool]:
        return {'r1': True, 'r2': True}

    def _get_tasks_modes(self) -> Dict[int, Dict[int, ModeConsumption]]:
        return {
            1: {1: ConstantModeConsumption({'r1': 0, 'r2': 0})},
            2: {1: ConstantModeConsumption({'r1': 1, 'r2': 1}), 2: ConstantModeConsumption({'r1': 2, 'r2': 0})},
            3: {1: ConstantModeConsumption({'r1': 1, 'r2': 0}), 2: ConstantModeConsumption({'r1': 0, 'r2': 1})},
            4: {1: ConstantModeConsumption({'r1': 2, 'r2': 1}), 2: ConstantModeConsumption({'r1': 2, 'r2': 0})},
            5: {1: ConstantModeConsumption({'r1': 0, 'r2': 0})}
        }

    def _get_mode_costs(self) -> Dict[int, Dict[int, float]]:
        return {
            1: {1: 0.},
            2: {1: 1., 2: 2.},
            3: {1: 1., 2: 1.},
            4: {1: 0., 2: 1.},
            5: {1: 0.}
        }

    def _get_resource_cost_per_time_unit(self) -> Dict[str, float]:
        return {
            'r1': 1,
            'r2': 2
        }

    def _get_objectives(self) -> List[SchedulingObjectiveEnum]:
        return [SchedulingObjectiveEnum.COST]

    def __init__(self):
        self.initialize_domain()

    def _get_max_horizon(self) -> int:
        return 50

    def _get_successors(self) -> Dict[int, List[int]]:
        return {1: [2,3], 2:[4], 3:[5], 4:[5], 5:[]}

    def _get_tasks_ids(self) -> Union[Set[int], Dict[int, Any], List[int]]:
        return set([1,2,3,4,5])

    def _get_resource_types_names(self) -> List[str]:
        return ['r1', 'r2']

    def _get_task_duration(self, task: int, mode: Optional[int] = 1, progress_from: Optional[float] = 0.) -> int:
        all_durations = {1: 0, 2: 5, 3: 6, 4: 4, 5: 0}
        return all_durations[task]

    def _get_original_quantity_resource(self, resource: str, **kwargs) -> int:
        all_resource_quantities = {'r1': 2, 'r2': 1}
        return all_resource_quantities[resource]


class ToyMS_RCPSPDomain(MultiModeMultiSkillRCPSP):
    def __init__(self):
        self.initialize_domain()

    def _get_resource_units_names(self) -> List[str]:
        return ["employee-1", "employee-2", "employee-3"]

    def _get_original_quantity_resource(self, resource: str, **kwargs) -> int:
        all_resource_quantities = {'r1': 2, 'r2': 1, "employee-1": 1, "employee-2": 1, "employee-3": 1}
        return all_resource_quantities[resource]

    def _get_max_horizon(self) -> int:
        return 50

    def _get_objectives(self) -> List[SchedulingObjectiveEnum]:
        return [SchedulingObjectiveEnum.MAKESPAN]

    def _get_successors(self) -> Dict[int, List[int]]:
        return {1: [2, 3], 2: [4], 3: [5], 4: [5], 5: []}

    def _get_tasks_ids(self) -> Union[Set[int], Dict[int, Any], List[int]]:
        return {1, 2, 3, 4, 5}

    def _get_tasks_modes(self) -> Dict[int, Dict[int, ModeConsumption]]:
        return {
            1: {1: ConstantModeConsumption({'r1': 0, 'r2': 0})},
            2: {1: ConstantModeConsumption({'r1': 1, 'r2': 1}), 2: ConstantModeConsumption({'r1': 2, 'r2': 0})},
            3: {1: ConstantModeConsumption({'r1': 1, 'r2': 0}), 2: ConstantModeConsumption({'r1': 0, 'r2': 1})},
            4: {1: ConstantModeConsumption({'r1': 2, 'r2': 1}), 2: ConstantModeConsumption({'r1': 2, 'r2': 0})},
            5: {1: ConstantModeConsumption({'r1': 0, 'r2': 0})}
        }

    def _get_task_duration(self, task: int, mode: Optional[int] = 1, progress_from: Optional[float] = 0.) -> int:
        all_durations = {1: 0, 2: 5, 3: 6, 4: 4, 5: 0}
        return all_durations[task]

    def _get_resource_types_names(self) -> List[str]:
        return ["r1", "r2"]

    def _get_resource_type_for_unit(self) -> Dict[str, str]:
        return None

    def _get_resource_renewability(self) -> Dict[str, bool]:
        return {"r1": True,
                "r2": True,
                "employee-1": True,
                "employee-2": True,
                "employee-3": True}

    def _get_all_resources_skills(self) -> Dict[str, Dict[str, Any]]:
        return {"employee-1": {"S1": 1},
                "employee-2": {"S2": 1},
                "employee-3": {"S3": 1},
                "r1": {},
                "r2": {}}

    def _get_all_tasks_skills(self) -> Dict[int, Dict[int, Dict[str, Any]]]:
        return {
            1: {1: {}},
            2: {1: {"S1": 1},
                2: {"S2": 1}},
            3: {1: {"S2": 1},
                2: {"S3": 1}},
            4: {1: {"S1": 1, "S2": 1},
                2: {"S2": 1, "S3": 1}},
            5: {1: {}}
        }


class ToySRCPSPDomain(SingleModeRCPSP_Stochastic_Durations):

    def _get_task_duration_distribution(self, task: int, mode: Optional[int] = 1,
                                        progress_from: Optional[float] = 0.,
                                        multivariate_settings: Optional[Dict[str, int]] = None) -> Distribution:
        all_distributions = {}
        all_distributions[1] = DiscreteDistribution([(0, 1.)])
        all_distributions[2] = DiscreteDistribution([(4, 0.25), (5, 0.5), (6, 0.25)])
        all_distributions[3] = DiscreteDistribution([(5, 0.25), (6, 0.5), (7, 0.25)])
        all_distributions[4] = DiscreteDistribution([(3, 0.5), (4, 0.5)])
        all_distributions[5] = DiscreteDistribution([(0, 1.)])

        return all_distributions[task]

    def _get_objectives(self) -> List[SchedulingObjectiveEnum]:
        return [SchedulingObjectiveEnum.MAKESPAN]

    def __init__(self):
        self.initialize_domain()

    def _get_max_horizon(self) -> int:
        return 20

    def _get_successors(self) -> Dict[int, List[int]]:
        return {1: [2,3], 2: [4], 3: [5], 4: [5], 5: []}

    def _get_tasks_ids(self) -> Union[Set[int], Dict[int, Any], List[int]]:
        return set([1,2,3,4,5])

    def _get_tasks_mode(self) -> Dict[int, ModeConsumption]:
        return {
            1: ConstantModeConsumption({'r1': 0, 'r2': 0}),
            2: ConstantModeConsumption({'r1': 1, 'r2': 1}),
            3: ConstantModeConsumption({'r1': 1, 'r2': 0}),
            4: ConstantModeConsumption({'r1': 2, 'r2': 1}),
            5: ConstantModeConsumption({'r1': 0, 'r2': 0})
        }

    def _get_resource_types_names(self) -> List[str]:
        return ['r1', 'r2']

    def _get_original_quantity_resource(self, resource: str, **kwargs) -> int:
        all_resource_quantities = {'r1': 2, 'r2': 1}
        return all_resource_quantities[resource]


class ConditionElementsExample(Enum):
    OK = 0
    PROBLEM_OPERATION_2 = 1
    PROBLEM_OPERATION_3 = 2


class ToyCondSRCPSPDomain(SingleModeRCPSP_Stochastic_Durations_WithConditionalTasks):

    def _get_all_condition_items(self) -> Enum:
         return ConditionElementsExample

    def _get_task_on_completion_added_conditions(self) -> Dict[int, List[Distribution]]:
        completion_conditions_dict = {}

        completion_conditions_dict[2] = [
                    DiscreteDistribution(
                        [(ConditionElementsExample.PROBLEM_OPERATION_2, 0.1), (ConditionElementsExample.OK, 0.9)])
                ]
        completion_conditions_dict[3] = [
            DiscreteDistribution(
                [(ConditionElementsExample.PROBLEM_OPERATION_3, 0.9), (ConditionElementsExample.OK, 0.1)])
        ]

        return completion_conditions_dict

    def _get_task_existence_conditions(self) -> Dict[int, List[int]]:
        existence_conditions_dict = {
            5: [self.get_all_condition_items().PROBLEM_OPERATION_2],
            6: [self.get_all_condition_items().PROBLEM_OPERATION_3]
        }
        return existence_conditions_dict

    def _get_task_duration_distribution(self, task: int, mode: Optional[int] = 1,
                                       progress_from: Optional[float] = 0.,
                                        multivariate_settings: Optional[Dict[str, int]] = None) -> Distribution:
        all_distributions = {}
        all_distributions[1] = DiscreteDistribution([(0, 1.)])
        all_distributions[2] = DiscreteDistribution([(4, 0.25), (5, 0.5), (6, 0.25)])
        all_distributions[3] = DiscreteDistribution([(5, 0.25), (6, 0.5), (7, 0.25)])
        all_distributions[4] = DiscreteDistribution([(3, 0.5), (4, 0.5)])
        all_distributions[5] = DiscreteDistribution([(4, 0.5), (5, 0.5)])
        all_distributions[6] = DiscreteDistribution([(2, 0.5), (3, 0.5)])
        all_distributions[7] = DiscreteDistribution([(0, 1.)])

        return all_distributions[task]

    def _get_objectives(self) -> List[SchedulingObjectiveEnum]:
        return [SchedulingObjectiveEnum.MAKESPAN]

    def __init__(self):
        self.initialize_domain()

    def _get_max_horizon(self) -> int:
        return 50

    def _get_successors(self) -> Dict[int, List[int]]:
        return {1: [2,3], 2:[4], 3:[7], 4:[7], 5:[7], 6:[7], 7:[]}

    def _get_tasks_ids(self) -> Union[Set[int], Dict[int, Any], List[int]]:
        return set([1,2,3,4,5,6,7])

    def _get_tasks_mode(self) -> Dict[int, ModeConsumption]:
        return {
            1: ConstantModeConsumption({'r1': 0, 'r2': 0}),
            2: ConstantModeConsumption({'r1': 1, 'r2': 1}),
            3: ConstantModeConsumption({'r1': 1, 'r2': 0}),
            4: ConstantModeConsumption({'r1': 2, 'r2': 1}),
            5: ConstantModeConsumption({'r1': 0, 'r2': 1}),
            6: ConstantModeConsumption({'r1': 1, 'r2': 0}),
            7: ConstantModeConsumption({'r1': 0, 'r2': 0})
        }

    def _get_resource_types_names(self) -> List[str]:
        return ['r1', 'r2']

    def _get_original_quantity_resource(self, resource: str, **kwargs) -> int:
        all_resource_quantities = {'r1': 2, 'r2': 1}
        return all_resource_quantities[resource]


class ToySimulatedCondSRCPSPDomain(SingleModeRCPSP_Simulated_Stochastic_Durations_WithConditionalTasks):

    def _get_all_condition_items(self) -> Enum:
         return ConditionElementsExample

    def _get_task_on_completion_added_conditions(self) -> Dict[int, List[Distribution]]:
        completion_conditions_dict = {}

        completion_conditions_dict[2] = [
                    DiscreteDistribution(
                        [(ConditionElementsExample.PROBLEM_OPERATION_2, 0.1), (ConditionElementsExample.OK, 0.9)])
                ]
        completion_conditions_dict[3] = [
            DiscreteDistribution(
                [(ConditionElementsExample.PROBLEM_OPERATION_3, 0.9), (ConditionElementsExample.OK, 0.1)])
        ]

        return completion_conditions_dict

    def _get_task_existence_conditions(self) -> Dict[int, List[int]]:
        existence_conditions_dict = {
            5: [self.get_all_condition_items().PROBLEM_OPERATION_2],
            6: [self.get_all_condition_items().PROBLEM_OPERATION_3]
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
        return {1, 2, 3, 4, 5, 6, 7}

    def _get_tasks_mode(self) -> Dict[int, ModeConsumption]:
        return {
            1: ConstantModeConsumption({'r1': 0, 'r2': 0}),
            2: ConstantModeConsumption({'r1': 1, 'r2': 1}),
            3: ConstantModeConsumption({'r1': 1, 'r2': 0}),
            4: ConstantModeConsumption({'r1': 2, 'r2': 1}),
            5: ConstantModeConsumption({'r1': 0, 'r2': 1}),
            6: ConstantModeConsumption({'r1': 1, 'r2': 0}),
            7: ConstantModeConsumption({'r1': 0, 'r2': 0})
        }

    def _get_resource_types_names(self) -> List[str]:
        return ['r1', 'r2']

    def _get_original_quantity_resource(self, resource: str, **kwargs) -> int:
        all_resource_quantities = {'r1': 2, 'r2': 1}
        return all_resource_quantities[resource]

    def _sample_task_duration(self, task: int, mode: Optional[int] = 1, progress_from: Optional[float]=0.) -> int:
        val = random.randint(3, 6)
        return val


@pytest.mark.parametrize("domain", [
    (ToyRCPSPDomain()),
    (ToyMRCPSPDomain_WithCost()),
    (ToySRCPSPDomain()),
    (ToyMS_RCPSPDomain()),
    (ToyCondSRCPSPDomain()),
    (ToySimulatedCondSRCPSPDomain())
])
def test_rollout(domain):
    state = domain.get_initial_state()
    states, actions, values = rollout_episode(domain=domain,
                                              max_steps=1000,
                                              solver=None,
                                              from_memory=state,
                                              outcome_formatter=lambda o: f'{o.observation} - cost: {o.value.cost:.2f}')
    check_rollout_consistency(domain, states)


def check_rollout_consistency(domain, states: List[State]):
    check_precedence(domain, states)
    check_task_duration(domain, states)
    check_resource_constraints(domain, states)
    check_skills(domain, states)


def check_precedence(domain, states: List[State]):
    # Check precedence
    for id in domain.get_tasks_ids():
        if id in states[-1].tasks_complete:  # needed for conditional tasks
            start_1 = states[-1].tasks_details[id].start
            for pred_id in domain.get_predecessors_task(id):
                if pred_id in states[-1].tasks_complete:  # needed for conditional tasks
                    end_0 = states[-1].tasks_details[pred_id].end
                    assert start_1 >= end_0, 'precedence constraints not respecetd between tasks ' + str(
                        id) + ' and ' + str(pred_id)


def check_task_duration(domain, states: List[State]):
    # Check task durations on deterministic domains
    if isinstance(domain, DeterministicTaskDuration):
        for id in domain.get_tasks_ids():
            if id in states[-1].tasks_complete:  # needed for conditional tasks
                expected_duration = domain.get_task_duration(id, states[-1].tasks_mode[id])
                actual_duration = states[-1].tasks_details[id].end - states[-1].tasks_details[id].start
                assert actual_duration == expected_duration, 'duration different than expected for task ' + str(id)
    else:
        for id in domain.get_tasks_ids():
            if id in states[-1].tasks_complete:  # needed for conditional tasks
                expected_duration = states[-1].tasks_details[id].sampled_duration
                actual_duration = states[-1].tasks_details[id].end - states[-1].tasks_details[id].start
                assert actual_duration == expected_duration, 'duration different than expected for task ' + str(id)


def check_resource_constraints(domain, states: List[State]):
    # Check resource constraints on deterministic non-preemtive domains
    if isinstance(domain, DeterministicResourceAvailabilityChanges) \
            and isinstance(domain,
                           WithoutPreemptivity) and isinstance(domain, WithoutConditionalTasks):
        for t in range(states[-1].t):
            for res in domain.get_resource_types_names():
                total_available = domain.get_quantity_resource(res, t)
                total_consumed = 0
                for id in domain.get_tasks_ids():
                    if states[-1].tasks_details[id].start <= t and states[-1].tasks_details[id].end > t:
                        total_consumed += domain.get_task_consumption(id, states[-1].tasks_mode[id], res, t)
                assert total_consumed <= total_available, 'over consumption at t=' + str(t) + ' for res ' + res

            for res in domain.get_resource_units_names():
                total_available = domain.get_quantity_resource(res, t)
                total_consumed = 0
                for id in domain.get_tasks_ids():
                    if states[-1].tasks_details[id].start <= t and states[-1].tasks_details[id].end > t:
                        total_consumed += domain.get_task_consumption(id, states[-1].tasks_mode[id], res, t)
                assert total_consumed <= total_available, 'over consumption at t=' + str(t) + ' for res ' + res


def check_skills(domain, states: List[State]):
    if isinstance(domain, ToyMS_RCPSPDomain):
        ressource_units = domain.get_resource_units_names()
        for state in states:
            from skdecide.builders.scheduling.scheduling_domains import State
            st: State = state
            task_checked = set()
            for task in st.tasks_ongoing:
                if task in task_checked:
                    continue
                skill_asked = domain.get_skills_of_task(task=task, mode=st.tasks_mode[task])
                if len(skill_asked) > 0:  # this task require some skills !
                    res_unit_used = [r for r in st.resource_used_for_task[task] if r in ressource_units]
                    skills = {}
                    for res_unit in res_unit_used:
                        skills_of_ressource = domain.get_skills_of_resource(res_unit)
                        for s in skills_of_ressource:
                            if s not in skills:
                                skills[s] = 0
                            skills[s] += skills_of_ressource[s]
                    assert all(skills.get(s, 0) >= skill_asked[s] for s in skill_asked)
                task_checked.add(task)


@pytest.mark.parametrize("domain", [
    # (ToyRCPSPDomain()),
    # (ToyMRCPSPDomain_WithCost()),
    (ToyMS_RCPSPDomain())
])
@pytest.mark.parametrize("do_solver", [
    # (SolvingMethod.PILE),
    # (SolvingMethod.CP),
    #(SolvingMethod.LP), # Runs on Popo but not ORC
    #(SolvingMethod.LNS_CP), # long to run ...
    (SolvingMethod.GA),
])
def test_do(domain, do_solver):
    print('domain: ', domain)
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
                                              max_steps=1000,
                                              solver=solver,
                                              from_memory=state,
                                              action_formatter=lambda o: str(o),
                                              outcome_formatter=lambda o: f'{o.observation} - cost: {o.value.cost:.2f}')
    check_rollout_consistency(domain, states)

@pytest.mark.parametrize("domain_multiskill", [
    (ToyMS_RCPSPDomain()),
])
@pytest.mark.parametrize("do_solver_multiskill", [
    # (SolvingMethod.CP),
    # (SolvingMethod.LS),
    # (SolvingMethod.LP), # Runs on Popo but not ORC
    # (SolvingMethod.LNS_CP), # long to run ...
    (SolvingMethod.GA)
])
def test_do_mskill(domain_multiskill, do_solver_multiskill):
    domain_multiskill.set_inplace_environment(False)
    state = domain_multiskill.get_initial_state()
    print("Initial state : ", state)
    solver = DOSolver(policy_method_params=PolicyMethodParams(base_policy_method=BasePolicyMethod.SGS_PRECEDENCE,
                                                              delta_index_freedom=0,
                                                              delta_time_freedom=0),
                      method=do_solver_multiskill)
    solver.solve(domain_factory=lambda: domain_multiskill)
    print(do_solver_multiskill)
    states, actions, values = rollout_episode(domain=domain_multiskill,
                                              max_steps=1000,
                                              solver=solver,
                                              from_memory=state,
                                              action_formatter=lambda o: str(o),
                                              outcome_formatter=lambda o: f'{o.observation} - cost: {o.value.cost:.2f}')
    check_rollout_consistency(domain_multiskill, states)


@pytest.mark.parametrize("domain", [
    # (ToyRCPSPDomain()),
    # (ToyMRCPSPDomain_WithCost()),
])
@pytest.mark.parametrize("solver_str", [
    ("LazyAstar"),
])
def test_planning_algos(domain, solver_str):
    domain.set_inplace_environment(False)
    state = domain.get_initial_state()
    print("Initial state : ", state)
    if solver_str == 'LazyAstar':
        solver = LazyAstar(from_state=state, heuristic=None, verbose=True)

    solver.solve(domain_factory=lambda: domain)
    states, actions, values = rollout_episode(domain=domain,
                                              max_steps=1000,
                                              solver=solver,
                                              from_memory=state,
                                              action_formatter=lambda o: str(o),
                                              outcome_formatter=lambda o: f'{o.observation} - cost: {o.value.cost:.2f}')
    check_rollout_consistency(domain, states)


@pytest.mark.parametrize("domain", [
    (ToyCondSRCPSPDomain()),
])
def test_conditional_task_models(domain):
    n_rollout = 250
    counters = {'PROBLEM_OPERATION_2': 0, 'PROBLEM_OPERATION_3': 0}
    domain.set_inplace_environment(False)
    for i in range(n_rollout):
        state = domain.get_initial_state()
        states, actions, values = rollout_episode(domain=domain,
                                                  max_steps=1000,
                                                  solver=None,
                                                  from_memory=state.copy(),
                                                  outcome_formatter=None)
                                              # outcome_formatter=lambda o: f'{o.observation} - cost: {o.value.cost:.2f}')
        if ConditionElementsExample.PROBLEM_OPERATION_2 in states[-1]._current_conditions:
            counters['PROBLEM_OPERATION_2'] += 1
        if ConditionElementsExample.PROBLEM_OPERATION_3 in states[-1]._current_conditions:
            counters['PROBLEM_OPERATION_3'] += 1
    counters['PROBLEM_OPERATION_2'] = float(counters['PROBLEM_OPERATION_2']) / float(n_rollout)
    counters['PROBLEM_OPERATION_3'] = float(counters['PROBLEM_OPERATION_3']) / float(n_rollout)
    print('counters:', counters)
    assert 0.05 <= counters['PROBLEM_OPERATION_2'] <= 0.15
    assert 0.85 <= counters['PROBLEM_OPERATION_3'] <= 0.95


@pytest.mark.parametrize("domain", [
    (ToyRCPSPDomain()),
    (ToyMS_RCPSPDomain())
])
@pytest.mark.parametrize("do_solver", [
    (SolvingMethod.CP),
    (SolvingMethod.LP)
])
def test_optimality(domain, do_solver):
    print('domain: ', domain)
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
                                              max_steps=1000,
                                              solver=solver,
                                              from_memory=state,
                                              action_formatter=lambda o: str(o),
                                              outcome_formatter=lambda o: f'{o.observation} - cost: {o.value.cost:.2f}')

    if isinstance(domain, ToyRCPSPDomain):
        makespan = max([states[-1].tasks_details[x].end for x in states[-1].tasks_complete])
        assert makespan == optimal_solutions['ToyRCPSPDomain']['makespan']
    if isinstance(domain, ToyMS_RCPSPDomain):
        makespan = max([states[-1].tasks_details[x].end for x in states[-1].tasks_complete])
        assert makespan == optimal_solutions['ToyMS_RCPSPDomain']['makespan']

@pytest.mark.parametrize("domain", [
    (ToySRCPSPDomain())
])
def test_compute_all_graph(domain):
    # Check also examples/scheduling/pi2/rcpsp_pi2/check_uncertain_domain to maybe check if the DFSExploration
    # create well the graph.
    from itertools import count
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
                              max_edges=2000,
                              score_function=score_state,
                              max_nodes=2000)
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
                    assert duration_task_for_ns in task_duration and \
                           prob == task_duration[duration_task_for_ns]  # duration are coherent with the input distribution


def test_basic():
    domain_rcpsp = load_domain("j1201_1.sm")
    domain_mrcpsp = load_domain("j1010_2.mm")
    assert isinstance(domain_rcpsp, RCPSP)
    assert isinstance(domain_mrcpsp, MRCPSP)
    # domain: RCPSP = load_domain("j1201_1.sm")
    state: State = domain_rcpsp.get_initial_state()
    print("Initial state : ", state)
    assert len(state.tasks_ongoing) == 0
    assert len(state.tasks_complete) == 0
    assert len(state.tasks_paused) == 0
    actions = domain_rcpsp.get_applicable_actions(state)
    action_list: List[SchedulingAction] = actions.get_elements()
    assert len(action_list) == 2
    action_start_source = [ac for ac in action_list
                           if ac.action == SchedulingActionEnum.START
                           and ac.task == 1]
    assert len(action_start_source) == 1
    next_state = domain_rcpsp.get_next_state(state, action_start_source[0])
    print("New state ", next_state)
    assert len(next_state.tasks_complete) == 1  # it is a dummy task, it should be completed now.

    action_list = domain_rcpsp.get_applicable_actions(state).get_elements()

    states, actions, values = rollout_episode(domain=domain_rcpsp,
                                              solver=None,
                                              from_memory=state,
                                              max_steps=500,
                                              outcome_formatter=lambda o: f'{o.observation} - cost: {o.value.cost:.2f}')
    print("rollout done")
    print('end times: ')
    for task_id in states[-1].tasks_details.keys():
        print('end task', task_id, ': ', states[-1].tasks_details[task_id].end)


@pytest.mark.parametrize("domain", [
    (ToySRCPSPDomain())
])
def test_sgs_policies(domain):
    deterministic_domains = build_n_determinist_from_stochastic(domain,
                                                                nb_instance=1)
    training_domains = deterministic_domains
    training_domains_names = ["my_toy_domain"]

    domain.set_inplace_environment(False)
    state = domain.get_initial_state()

    # Using a stochastic domain as reference + executing on stochastic domain
    solver = GPHH(training_domains=training_domains,
                  domain_model=domain,
                  weight=-1,
                  verbose=True,
                  training_domains_names=training_domains_names,
                  params_gphh=ParametersGPHH.fast_test()
                  )
    solver.solve(domain_factory=lambda: domain)
    solver.set_domain(domain)
    states, actions, values = rollout_episode(domain=domain,
                                              max_steps=1000,
                                              solver=solver,
                                              from_memory=state,
                                              verbose=False,
                                              outcome_formatter=lambda
                                                  o: f'{o.observation} - cost: {o.value.cost:.2f}')
    print("Cost :", sum([v.cost for v in values]))
    check_rollout_consistency(domain, states)

    # Using a deterministic domain as reference + executing on deterministic domain
    solver = GPHH(training_domains=training_domains,
                  domain_model=training_domains[0],
                  weight=-1,
                  verbose=True,
                  training_domains_names=training_domains_names,
                  params_gphh=ParametersGPHH.fast_test()
                  )
    solver.solve(domain_factory=lambda: training_domains[0])
    solver.set_domain(training_domains[0])
    states, actions, values = rollout_episode(domain=training_domains[0],
                                              max_steps=1000,
                                              solver=solver,
                                              from_memory=state,
                                              verbose=False,
                                              outcome_formatter=lambda
                                                  o: f'{o.observation} - cost: {o.value.cost:.2f}')
    print("Cost :", sum([v.cost for v in values]))
    check_rollout_consistency(domain, states)
