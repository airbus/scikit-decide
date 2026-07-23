from collections import defaultdict
from typing import Any, Collection, Optional

from discrete_optimization.generic_tasks_tools.calendar_resource import (
    consolidate_availability_intervals,
)

from skdecide.builders.domain.scheduling.conditional_tasks import (
    WithoutConditionalTasks,
)
from skdecide.builders.domain.scheduling.modes import (
    ConstantModeConsumption,
    ModeConsumption,
)
from skdecide.builders.domain.scheduling.preallocations import WithoutPreallocations
from skdecide.builders.domain.scheduling.preemptivity import WithoutPreemptivity
from skdecide.builders.domain.scheduling.resource_availability import (
    DeterministicResourceAvailabilityChanges,
)
from skdecide.builders.domain.scheduling.resource_consumption import (
    ConstantResourceConsumption,
)
from skdecide.builders.domain.scheduling.scheduling_domains import (
    DeterministicSchedulingDomain,
    SchedulingObjectiveEnum,
)
from skdecide.builders.domain.scheduling.task_duration import DeterministicTaskDuration
from skdecide.builders.domain.scheduling.task_progress import DeterministicTaskProgress
from skdecide.builders.domain.scheduling.time_lag import TimeLag
from skdecide.builders.domain.scheduling.time_windows import TimeWindow

# types for annotations
Task = int
NonSkillCumulativeResource = str
NonRenewableResource = str
Skill = str
CumulativeResource = NonSkillCumulativeResource | Skill
UnaryResource = str
UnaryAvailabilityIntervals = list[tuple[int, int]]  # start, end
AvailabilityIntervals = list[tuple[int, int, int]]  # start, end, value


class AsDOSchedulingDomain(
    DeterministicSchedulingDomain,
    ConstantResourceConsumption,
    WithoutPreemptivity,
    DeterministicTaskDuration,
    DeterministicTaskProgress,
    WithoutPreallocations,
    DeterministicResourceAvailabilityChanges,
    WithoutConditionalTasks,
):
    """Equivalent to discrete-optimization generic scheduling domain."""

    def __init__(
        self,
        horizon: int,
        durations_per_mode: dict[Task, dict[int, int]],
        resource_consumptions: Optional[
            dict[Task, dict[int, dict[CumulativeResource | NonRenewableResource, int]]]
        ] = None,
        successors: Optional[dict[Task, list[Task]]] = None,
        unary_resources: Optional[list[UnaryResource]] = None,
        unary_resources_skills: Optional[dict[UnaryResource, dict[Skill, int]]] = None,
        unary_resources_availabilities: Optional[
            dict[UnaryResource, UnaryAvailabilityIntervals]
        ] = None,
        skills: Optional[set[Skill]] = None,
        non_skill_cumulative_resources: Optional[
            dict[CumulativeResource, int | AvailabilityIntervals]
        ] = None,
        non_renewable_resources: Optional[dict[NonRenewableResource, int]] = None,
        time_windows: Optional[
            dict[Task, tuple[int | None, int | None, int | None, int | None]]
        ] = None,
        end_to_start_min_time_lags: Optional[list[tuple[Task, Task, int]]] = None,
        end_to_start_max_time_lags: Optional[list[tuple[Task, Task, int]]] = None,
        objective: list[SchedulingObjectiveEnum]
        | SchedulingObjectiveEnum = SchedulingObjectiveEnum.MAKESPAN,
        mode_costs: Optional[dict[Task, dict[int, int]]] = None,
        unary_resource_costs: Optional[
            dict[Task, dict[int, dict[UnaryResource, int]]]
        ] = None,
    ):
        """

        # Parameters
        horizon: max allowed time to finish the tasks
        durations_per_mode: task -> mode -> duration. Tasks durations, mode by mode.
            This is used to know all available tasks, all available modes for a given task, and corresponding durations.
        resource_consumptions: task -> mode -> resource -> conso.
            Cumulative (including skills) or non-renewable resource consumption, task by task, mode by mode. The resource can be a skill.
            Missing key => conso = 0
        successors: maps a task to its successors in the precedence graph.
            Each successor task must start after the given task ends.
            Default to no precedence constraints. Note that a consolidated version of it will
            be constructed using the time lags constraints.
        unary_resources: available unary resources.  Default to none.
        unary_resources_skills: skill values of each unary resource. Mssing key => skill value = 0
        unary_resources_availabilities: availability of unary resources on the form of list of intervals (start, end)
            Missing key => always available.
        skills: available skills
        non_skill_cumulative_resources: cumulative resources (excluding skills) availabilities.
            Format: either int => always available at the given max capacity,
            or list of intervals + capacity (start, end, value)
        non_renewable_resources: non-renewable resources max capacities
        time_windows: maps task to start_lb, end_lb, start_ub, end_ub s.t.
            start_lb <= start(task) <= start_ub and end_lb <= end(task) <= end_ub
            missing or none value means 0 (lb) or self.horizon (ub)
        end_to_start_min_time_lags: min time lags constraints between first task end and second task start.
            task1, task2, offset meaning end(task1) + offset <= start(task2)
        end_to_start_max_time_lags: max time lags constraints between first task end and second task start.
            task1, task2, offset meaning end(task1) + offset >= start(task2)
        objective: objective for the domain. Default to minimization of makespan.
            Either a single objective which leads to the computation to corresponding transition value,
            or a list of objectives so thatthe transition value is the sum of values corresponding to each objective.
        unary_resource_costs: cost of allocating each unary resource. Missing key => cost = 0.
        """
        self.horizon = horizon
        self.durations_per_mode = durations_per_mode

        # default values
        if resource_consumptions is None:
            self.resource_consumptions: dict[
                Task, dict[int, dict[CumulativeResource | NonRenewableResource, int]]
            ] = {}
        else:
            self.resource_consumptions = resource_consumptions
        if successors is None:
            self.successors: dict[Task, list[Task]] = {}
        else:
            self.successors = successors
        if unary_resources is None:
            self.unary_resources: list[UnaryResource] = list()
        else:
            self.unary_resources = unary_resources
        if unary_resources_skills is None:
            self.unary_resources_skills: dict[UnaryResource, dict[Skill, int]] = {}
        else:
            self.unary_resources_skills = unary_resources_skills
        if unary_resources_availabilities is None:
            self.unary_resources_availabilities: dict[
                UnaryResource, UnaryAvailabilityIntervals
            ] = {}
        else:
            self.unary_resources_availabilities = unary_resources_availabilities
        self.unary_resources_consolidated_availabilities = {}
        if skills is None:
            self.skills: set[Skill] = set()
        else:
            self.skills = skills
        if non_skill_cumulative_resources is None:
            self.non_skill_cumulative_resources: dict[
                NonSkillCumulativeResource, int | AvailabilityIntervals
            ] = {}
        else:
            self.non_skill_cumulative_resources = non_skill_cumulative_resources
        self.non_skill_cumulative_resources_consolidated_availabilities = {}
        if non_renewable_resources is None:
            self.non_renewable_resources: dict[NonRenewableResource, int] = {}
        else:
            self.non_renewable_resources = non_renewable_resources
        if time_windows is None:
            self.time_windows: dict[
                Task, tuple[int | None, int | None, int | None, int | None]
            ] = {}
        else:
            self.time_windows = time_windows
        if end_to_start_min_time_lags is None:
            self.end_to_start_min_time_lags: list[tuple[Task, Task, int]] = []
        else:
            self.end_to_start_min_time_lags = end_to_start_min_time_lags
        if end_to_start_max_time_lags is None:
            self.end_to_start_max_time_lags: list[tuple[Task, Task, int]] = []
        else:
            self.end_to_start_max_time_lags = end_to_start_max_time_lags
        if mode_costs is None:
            self.mode_costs: dict[int, dict[int, int]] = {}
        else:
            self.mode_costs = mode_costs
        if unary_resource_costs is None:
            self.unary_resource_costs = {}
        else:
            self.unary_resource_costs = unary_resource_costs
        if isinstance(objective, SchedulingObjectiveEnum):
            self.objectives = [objective]
        else:
            self.objectives = objective

        self.initialize_domain()

    def _get_max_horizon(self) -> int:
        return self.horizon

    def _get_objectives(self) -> list[SchedulingObjectiveEnum]:
        return self.objectives

    def _get_successors(self) -> dict[int, list[int]]:
        return self.successors

    def _get_tasks_ids(self) -> Collection[int]:
        return set(self.durations_per_mode)

    def _get_tasks_modes(self) -> dict[int, dict[int, ModeConsumption]]:
        return {
            task: {
                mode: ConstantModeConsumption(
                    {
                        resource: conso
                        for resource, conso in mode_dict.items()
                        if resource not in self.skills
                    }
                )
                for mode, mode_dict in task_consumptions_per_mode.items()
            }
            for task, task_consumptions_per_mode in self.resource_consumptions.items()
        }

    def _get_resource_types_names(self) -> list[str]:
        return list(self.non_skill_cumulative_resources) + list(
            self.non_renewable_resources
        )

    def _get_resource_units_names(self) -> list[str]:
        return self.unary_resources

    def _get_resource_type_for_unit(self) -> dict[str, str]:
        return {}

    def _get_resource_renewability(self) -> dict[str, bool]:
        return {resource: False for resource in self.non_renewable_resources} | {
            resource: True for resource in self.non_skill_cumulative_resources
        }

    def _get_task_duration(
        self, task: int, mode: int = 1, progress_from: float = 0.0
    ) -> int:
        return self.durations_per_mode[task][mode]

    def _get_all_resources_skills(self) -> dict[str, dict[str, Any]]:
        return {
            unary_resource: {skill: value for skill, value in skills.items()}
            for unary_resource, skills in self.unary_resources_skills.items()
        }

    def _get_all_tasks_skills(self) -> dict[int, dict[int, dict[str, Any]]]:
        return {
            task: {
                mode: {
                    skill: conso
                    for skill, conso in mode_dict.items()
                    if skill in self.skills
                }
                for mode, mode_dict in task_consumptions_per_mode.items()
            }
            for task, task_consumptions_per_mode in self.resource_consumptions.items()
        }

    def _get_time_lags(self) -> dict[int, dict[int, TimeLag]]:
        min_time_lags_dict: dict[int, dict[int, Optional[int]]] = defaultdict(
            lambda: defaultdict(lambda: None)
        )
        max_time_lags_dict: dict[int, dict[int, Optional[int]]] = defaultdict(
            lambda: defaultdict(lambda: None)
        )
        for task1, task2, offset in self.end_to_start_min_time_lags:
            min_time_lags_dict[task1][task2] = offset
        for task1, task2, offset in self.end_to_start_max_time_lags:
            max_time_lags_dict[task1][task2] = offset
        return {
            task1: {
                task2: TimeLag(
                    minimum_time_lag=min_time_lags_dict[task1][task2],
                    maximum_time_lag=max_time_lags_dict[task1][task2],
                )
                for task2 in set(min_time_lags_dict[task1])
                | set(max_time_lags_dict[task1])
            }
            for task1 in set(min_time_lags_dict) | set(max_time_lags_dict)
        }

    def _get_time_window(self) -> dict[int, TimeWindow]:
        return {
            task: TimeWindow(
                earliest_start=_get_default_time_lower_bound(start_lb),
                earliest_end=_get_default_time_lower_bound(end_lb),
                latest_start=_get_default_time_upper_bound(
                    start_ub, horizon=self.horizon
                ),
                latest_end=_get_default_time_upper_bound(end_ub, horizon=self.horizon),
            )
            for task, (start_lb, end_lb, start_ub, end_ub) in self.time_windows.items()
        }

    def _get_quantity_resource(self, resource: str, time: int, **kwargs) -> int:
        if resource in self.non_renewable_resources:
            return self.non_renewable_resources[resource]
        elif resource in self.non_skill_cumulative_resources:
            availability = self.non_skill_cumulative_resources[resource]
            if isinstance(availability, int):
                return availability
            elif (
                resource
                not in self.non_skill_cumulative_resources_consolidated_availabilities
            ):
                self.non_skill_cumulative_resources_consolidated_availabilities[
                    resource
                ] = consolidate_availability_intervals(
                    intervals=availability,
                    horizon=self.horizon,
                )
            return _extract_quantitity_from_availability_intervals(
                availability=self.non_skill_cumulative_resources_consolidated_availabilities[
                    resource
                ],
                time=time,
            )
        elif resource in self.unary_resources:
            if resource not in self.unary_resources_availabilities:
                return 1
            elif resource not in self.unary_resources_consolidated_availabilities:
                self.unary_resources_consolidated_availabilities[resource] = (
                    consolidate_availability_intervals(
                        intervals=[
                            (start, end, 1)
                            for start, end in self.unary_resources_availabilities[
                                resource
                            ]
                        ],
                        horizon=self.horizon,
                    )
                )
            return _extract_quantitity_from_availability_intervals(
                availability=self.unary_resources_consolidated_availabilities[resource],
                time=time,
            )
        else:
            raise ValueError(f"Resource {resource} unknown.")

    def _get_mode_costs(self) -> dict[int, dict[int, float]]:
        return self.mode_costs

    def _get_resource_cost_per_time_unit(self) -> dict[str, float]:
        """Cost per time unit of a resource.

        Hypotheses:
        - cost only for unary resources to be allocated (other costs included in mode cost)
        - given unary resources costs are really proportional to time

        Disclaimer: To compute it, we take the first unary_resource cost found in `sef.unary_resource_costs` and divide it
        by the task duration. We do not check that each such cost give the same cost pet time unit.

        """
        resource_cost_per_time_unit = defaultdict(lambda: 0)
        for task, task_unary_resource_costs in self.unary_resource_costs.items():
            for (
                mode,
                task_mode_unary_resource_costs,
            ) in task_unary_resource_costs.items():
                for unary_resource, cost in task_mode_unary_resource_costs.items():
                    if unary_resource not in resource_cost_per_time_unit:
                        resource_cost_per_time_unit[unary_resource] = (
                            cost / self.durations_per_mode[task][mode]
                        )
        return resource_cost_per_time_unit


def _get_default_time_lower_bound(lb: int | None) -> int:
    if lb is None:
        return 0
    else:
        return lb


def _get_default_time_upper_bound(ub: int | None, horizon: int) -> int:
    if ub is None:
        return horizon
    else:
        return ub


def _extract_quantitity_from_availability_intervals(
    availability: AvailabilityIntervals, time: int
) -> int:
    """Extract the available quantity at given time.

    Hypothesis: availability is a partition of (0, horizon), ie
    - starts at 0
    - increasing starts
    - not intersecting intervals
    - end[i-1] == start[i]

    """
    for start, end, value in availability:
        if time < end:
            return value
    return 0
