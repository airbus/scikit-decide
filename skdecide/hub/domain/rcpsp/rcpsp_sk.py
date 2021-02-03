from typing import Optional, List, Dict, Union, Set, Any
from skdecide import TransitionValue, Distribution, DiscreteDistribution
from skdecide.builders.scheduling.modes import ModeConsumption, ConstantModeConsumption
from skdecide.builders.scheduling.scheduling_domains import MultiModeRCPSP, \
    SchedulingDomain, DeterministicSchedulingDomain, SingleModeRCPSP, SingleMode, \
    SingleModeRCPSP_Stochastic_Durations, MultiModeRCPSP_Stochastic_Durations, \
    MultiModeMultiSkillRCPSP, SchedulingObjectiveEnum, State, MultiModeMultiSkillRCPSPCalendar, MultiModeRCPSPCalendar, MultiModeRCPSPCalendar_Stochastic_Durations

from skdecide.builders.domain import DeterministicTransitions


class D(MultiModeRCPSP):
    pass


class MRCPSP(D):
    def __init__(self,
                 resource_names: List[str]=None,
                 task_ids: List[int]=None,
                 tasks_mode: Dict[int, Dict[int, Dict[str, int]]]=None,
                 successors: Dict[int, List[int]]=None,
                 max_horizon: int=None,
                 resource_availability: Dict[str, int]=None,
                 resource_renewable: Dict[str, bool]=None):
        self.resource_names = resource_names
        self.task_ids = task_ids
        self.tasks_mode = tasks_mode
        # transform the "mode_details" dict that we largely used in DO in the good format.
        self.task_mode_dict = {}
        self.duration_dict = {}
        for task in self.tasks_mode:
            self.task_mode_dict[task] = {}
            self.duration_dict[task] = {}
            for mode in self.tasks_mode[task]:
                self.task_mode_dict[task][mode] = ConstantModeConsumption({})
                for r in self.tasks_mode[task][mode]:
                    if r in self.resource_names:
                        self.task_mode_dict[task][mode].mode_details[r] = [self.tasks_mode[task][mode][r]]
                self.duration_dict[task][mode] = self.tasks_mode[task][mode]["duration"]
        self.successors = successors
        self.max_horizon = max_horizon
        self.resource_availability = resource_availability
        self.resource_renewable = resource_renewable
        self.initialize_domain()

    def _get_tasks_modes(self) -> Dict[int, Dict[int, ModeConsumption]]:
        return self.task_mode_dict

    def _get_resource_renewability(self) -> Dict[str, bool]:
        return self.resource_renewable

    def _get_max_horizon(self) -> int:
        return self.max_horizon

    def _get_successors(self) -> Dict[int, List[int]]:
        return self.successors

    def _get_tasks_ids(self) -> Union[Set[int], Dict[int, Any], List[int]]:
        return self.task_ids

    def _get_task_duration(self, task: int, mode: Optional[int] = 1, progress_from: Optional[float] = 0.) -> int:
        return self.duration_dict[task][mode]

    def _get_original_quantity_resource(self, resource: str, **kwargs) -> int:
        return self.resource_availability[resource]

    def _get_resource_types_names(self) -> List[str]:
        return self.resource_names

    def _get_objectives(self) -> List[int]:
        return [SchedulingObjectiveEnum.MAKESPAN]


class D(MultiModeRCPSPCalendar):
    pass


class MRCPSPCalendar(D):
    def _get_quantity_resource(self, resource: str, time: int, **kwargs) -> int:
        return self.resource_availability[resource][time]

    def __init__(self,
                 resource_names: List[str]=None,
                 task_ids: List[int]=None,
                 tasks_mode: Dict[int, Dict[int, Dict[str, int]]]=None,
                 successors: Dict[int, List[int]]=None,
                 max_horizon: int=None,
                 resource_availability: Dict[str, List[int]]=None,
                 resource_renewable: Dict[str, bool]=None):
        self.resource_names = resource_names
        self.task_ids = task_ids
        self.tasks_mode = tasks_mode
        # transform the "mode_details" dict that we largely used in DO in the good format.
        self.task_mode_dict = {}
        self.duration_dict = {}
        for task in self.tasks_mode:
            self.task_mode_dict[task] = {}
            self.duration_dict[task] = {}
            for mode in self.tasks_mode[task]:
                self.task_mode_dict[task][mode] = ConstantModeConsumption({})
                for r in self.tasks_mode[task][mode]:
                    if r in self.resource_names:
                        self.task_mode_dict[task][mode].mode_details[r] = [self.tasks_mode[task][mode][r]]
                self.duration_dict[task][mode] = self.tasks_mode[task][mode]["duration"]
        self.successors = successors
        self.max_horizon = max_horizon
        self.resource_availability = resource_availability
        self.original_resource_availability = {r: max(self.resource_availability[r])
                                               for r in self.resource_availability}

        self.resource_renewable = resource_renewable
        self.initialize_domain()

    def _get_tasks_modes(self) -> Dict[int, Dict[int, ModeConsumption]]:
        return self.task_mode_dict

    def _get_resource_renewability(self) -> Dict[str, bool]:
        return self.resource_renewable

    def _get_max_horizon(self) -> int:
        return self.max_horizon

    def _get_successors(self) -> Dict[int, List[int]]:
        return self.successors

    def _get_tasks_ids(self) -> Union[Set[int], Dict[int, Any], List[int]]:
        return self.task_ids

    def _get_task_duration(self, task: int, mode: Optional[int] = 1, progress_from: Optional[float] = 0.) -> int:
        return self.duration_dict[task][mode]

    def _get_original_quantity_resource(self, resource: str, **kwargs) -> int:
        return self.original_resource_availability[resource]

    def _get_resource_types_names(self) -> List[str]:
        return self.resource_names

    def _get_objectives(self) -> List[int]:
        return [SchedulingObjectiveEnum.MAKESPAN]


class RCPSP(MRCPSP, SingleMode):
    def __init__(self,
                 resource_names: List[str]=None,
                 task_ids: List[int]=None,
                 tasks_mode: Dict[int, Dict[int, Dict[str, int]]]=None,
                 successors: Dict[int, List[int]]=None,
                 max_horizon: int=None,
                 resource_availability: Dict[str, int]=None,
                 resource_renewable: Dict[str, bool]=None):
        MRCPSP.__init__(self, resource_names=resource_names, task_ids=task_ids,
                        tasks_mode=tasks_mode, successors=successors,
                        max_horizon=max_horizon, resource_availability=resource_availability,
                        resource_renewable=resource_renewable)
        self.tasks_modes_rcpsp = {t: self.task_mode_dict[t][1]
                                  for t in self.task_mode_dict}

    def _get_tasks_mode(self) -> Dict[int, ModeConsumption]:
        return self.tasks_modes_rcpsp


class RCPSPCalendar(MRCPSPCalendar, SingleMode):
    def __init__(self,
                 resource_names: List[str]=None,
                 task_ids: List[int]=None,
                 tasks_mode: Dict[int, Dict[int, Dict[str, int]]]=None,
                 successors: Dict[int, List[int]]=None,
                 max_horizon: int=None,
                 resource_availability: Dict[str, List[int]]=None,
                 resource_renewable: Dict[str, bool]=None):
        MRCPSPCalendar.__init__(self, resource_names=resource_names, task_ids=task_ids,
                                tasks_mode=tasks_mode, successors=successors,
                                max_horizon=max_horizon, resource_availability=resource_availability,
                                resource_renewable=resource_renewable)
        self.tasks_modes_rcpsp = {t: self.task_mode_dict[t][1]
                                  for t in self.task_mode_dict}

    def _get_tasks_mode(self) -> Dict[int, ModeConsumption]:
        return self.tasks_modes_rcpsp


class Stochastic_RCPSP(MultiModeRCPSP_Stochastic_Durations):
    def _get_max_horizon(self) -> int:
        return self.max_horizon

    def _get_objectives(self) -> List[SchedulingObjectiveEnum]:
        return [SchedulingObjectiveEnum.MAKESPAN]

    def _get_successors(self) -> Dict[int, List[int]]:
        return self.successors

    def _get_tasks_ids(self) -> Union[Set[int], Dict[int, Any], List[int]]:
        return self.task_ids

    def _get_tasks_modes(self) -> Dict[int, Dict[int, ModeConsumption]]:
        return self.task_mode_dict

    def _get_task_duration_distribution(self, task: int, mode: Optional[int] = 1,
                                        progress_from: Optional[float] = 0.,
                                        multivariate_settings: Optional[Dict[str, int]]=None) -> Distribution:
        return self.duration_distribution[task][mode]

    def _get_original_quantity_resource(self, resource: str, **kwargs) -> int:
        return self.resource_availability[resource]

    def _get_resource_types_names(self) -> List[str]:
        return self.resource_names

    def __init__(self,
                 resource_names: List[str]=None,
                 task_ids: List[int]=None,
                 tasks_mode: Dict[int, Dict[int, ModeConsumption]]=None, # ressource
                 duration_distribution: Dict[int, Dict[int, DiscreteDistribution]]=None,
                 successors: Dict[int, List[int]]=None,
                 max_horizon: int=None,
                 resource_availability: Dict[str, int]=None,
                 resource_renewable: Dict[str, bool]=None):
        self.resource_names = resource_names
        self.task_ids = task_ids
        self.tasks_mode = tasks_mode
        # transform the "mode_details" dict that we largely used in DO in the good format.
        self.task_mode_dict = self.tasks_mode
        self.duration_distribution = duration_distribution
        self.successors = successors
        self.max_horizon = max_horizon
        self.resource_availability = resource_availability
        self.resource_renewable = resource_renewable
        self.initialize_domain()


def build_stochastic_from_deterministic(rcpsp: MRCPSP, task_to_noise: Set[int]=None):
    if task_to_noise is None:
        task_to_noise = set(rcpsp.get_tasks_ids())
    duration_distribution = {}
    for task_id in rcpsp.get_tasks_ids():
        duration_distribution[task_id] = {}
        for mode in rcpsp.get_task_modes(task_id=task_id):
            duration = rcpsp.get_task_duration(task=task_id, mode=mode)
            if duration == 0 or task_id not in task_to_noise:
                distrib = DiscreteDistribution(values=[(duration, 1)])
            else:
                n = 10
                distrib = DiscreteDistribution(values=[(max(1, duration+i), 1/(2*n+1))
                                                       for i in range(-n, n+1)])
            duration_distribution[task_id][mode] = distrib

    return Stochastic_RCPSP(resource_names=rcpsp.get_resource_types_names(),
                            task_ids=rcpsp.get_tasks_ids(),
                            tasks_mode=rcpsp.get_tasks_modes(),  # ressource
                            duration_distribution=duration_distribution,
                            successors=rcpsp.successors,
                            max_horizon=rcpsp.max_horizon*2,
                            resource_availability=rcpsp.resource_availability,
                            resource_renewable=rcpsp.resource_renewable)


def build_n_determinist_from_stochastic(srcpsp: Stochastic_RCPSP, nb_instance: int):
    instances = []
    for i in range(nb_instance):
        modes = srcpsp.get_tasks_modes()
        modes_for_rcpsp = {task: {mode: {r:
                                         modes[task][mode].get_resource_need_at_time(r, 0)
                                         for r in modes[task][mode].get_ressource_names()}
                           for mode in modes[task]} for task in modes}
        for t in modes_for_rcpsp:
            for m in modes_for_rcpsp[t]:
                duration = srcpsp.sample_task_duration(task=t, mode=m)
                modes_for_rcpsp[t][m]["duration"] = duration

        resource_availability_dict = {}
        for r in srcpsp.get_resource_types_names():
            resource_availability_dict[r] = srcpsp.get_original_quantity_resource(r)

        instances += [MRCPSP(resource_names=srcpsp.get_resource_types_names(),
                             task_ids=srcpsp.get_tasks_ids(),
                             tasks_mode=modes_for_rcpsp,  # ressource
                             successors=srcpsp.successors,
                             # max_horizon=srcpsp.max_horizon,
                             max_horizon=srcpsp.get_max_horizon(),
                             # resource_availability=srcpsp.resource_availability,
                             resource_availability=resource_availability_dict,
                             resource_renewable=srcpsp.get_resource_renewability()
                      # resource_renewable=srcpsp.resource_renewable
                             )]
    return instances



class D(MultiModeRCPSPCalendar_Stochastic_Durations):
    pass


class SMRCPSPCalendar(D):

    def _get_task_duration_distribution(self, task: int, mode: Optional[int] = 1, progress_from: Optional[float] = 0.,
                                        multivariate_settings: Optional[Dict[str, int]] = None) -> Distribution:
        return self.duration_distribution[task][mode]

    def _get_quantity_resource(self, resource: str, time: int, **kwargs) -> int:
        return self.resource_availability[resource][time]

    def __init__(self,
                 resource_names: List[str]=None,
                 task_ids: List[int]=None,
                 tasks_mode: Dict[int, Dict[int, Dict[str, int]]]=None,
                 successors: Dict[int, List[int]]=None,
                 duration_distribution: Dict[int, Dict[int, DiscreteDistribution]] = None,
                 max_horizon: int=None,
                 resource_availability: Dict[str, List[int]]=None,
                 resource_renewable: Dict[str, bool]=None):
        self.resource_names = resource_names
        self.task_ids = task_ids
        self.tasks_mode = tasks_mode
        # transform the "mode_details" dict that we largely used in DO in the good format.
        self.task_mode_dict = {}
        self.duration_dict = {}
        for task in self.tasks_mode:
            self.task_mode_dict[task] = {}
            self.duration_dict[task] = {}
            for mode in self.tasks_mode[task]:
                self.task_mode_dict[task][mode] = ConstantModeConsumption({})
                for r in self.tasks_mode[task][mode]:
                    if r in self.resource_names:
                        self.task_mode_dict[task][mode].mode_details[r] = [self.tasks_mode[task][mode][r]]
                self.duration_dict[task][mode] = self.tasks_mode[task][mode]["duration"]
        self.successors = successors
        self.max_horizon = max_horizon
        self.resource_availability = resource_availability
        self.resource_renewable = resource_renewable
        self.duration_distribution = duration_distribution
        self.initialize_domain()

    def _get_tasks_modes(self) -> Dict[int, Dict[int, ModeConsumption]]:
        return self.task_mode_dict

    def _get_resource_renewability(self) -> Dict[str, bool]:
        return self.resource_renewable

    def _get_max_horizon(self) -> int:
        return self.max_horizon

    def _get_successors(self) -> Dict[int, List[int]]:
        return self.successors

    def _get_tasks_ids(self) -> Union[Set[int], Dict[int, Any], List[int]]:
        return self.task_ids

    def _get_task_duration(self, task: int, mode: Optional[int] = 1, progress_from: Optional[float] = 0.) -> int:
        return self.duration_dict[task][mode]

    def _get_original_quantity_resource(self, resource: str, **kwargs) -> int:
        # return self.resource_availability[resource]
        return self.original_resource_availability[resource]

    def _get_resource_types_names(self) -> List[str]:
        return self.resource_names

    def _get_objectives(self) -> List[int]:
        return [SchedulingObjectiveEnum.MAKESPAN]


class D(MultiModeMultiSkillRCPSP):
    pass


class MSRCPSP(D):
    def __init__(self,
                 skills_names: List[str] = None,
                 resource_unit_names: List[str] = None,
                 resource_type_names: List[str] = None,
                 resource_skills: Dict[str, Dict[str, Any]]=None,
                 task_ids: List[int] = None,
                 tasks_mode: Dict[int, Dict[int, Dict[str, int]]] = None,
                 successors: Dict[int, List[int]] = None,
                 max_horizon: int = None,
                 resource_availability: Dict[str, int] = None,
                 resource_renewable: Dict[str, bool] = None):
        self.skills_set = set(skills_names)
        self.resource_unit_names = resource_unit_names
        self.resource_type_names = resource_type_names
        self.resource_skills = resource_skills
        self.task_ids = task_ids
        self.tasks_mode = tasks_mode
        # transform the "mode_details" dict that we largely used in DO in the good format.
        self.task_mode_dict = {}
        self.task_skills_dict = {}
        self.duration_dict = {}
        for task in self.tasks_mode:
            self.task_mode_dict[task] = {}
            self.task_skills_dict[task] = {}
            self.duration_dict[task] = {}
            for mode in self.tasks_mode[task]:
                self.task_mode_dict[task][mode] = ConstantModeConsumption({})
                self.task_skills_dict[task][mode] = {}
                for r in self.tasks_mode[task][mode]:
                    if r in self.resource_type_names:
                        self.task_mode_dict[task][mode].mode_details[r] = [self.tasks_mode[task][mode][r]]
                    if r in self.skills_set:
                        self.task_skills_dict[task][mode][r] = self.tasks_mode[task][mode][r]
                self.duration_dict[task][mode] = self.tasks_mode[task][mode]["duration"]
        self.successors = successors
        self.max_horizon = max_horizon
        self.resource_availability = resource_availability
        self.resource_renewable = resource_renewable
        self.initialize_domain()

    def _get_resource_units_names(self) -> List[str]:
        """Return the names (string) of all resource units as a list."""
        return self.resource_unit_names

    def _get_resource_types_names(self) -> List[str]:
        return self.resource_type_names

    def _get_resource_type_for_unit(self) -> Dict[str, str]:
        """Return a dictionary where the key is a resource unit name and the value a resource type name.
        An empty dictionary can be used if there are no resource unit matching a resource type."""
        return None

    def get_max_horizon(self) -> int:
        return self.max_horizon

    def _get_tasks_modes(self) -> Dict[int, Dict[int, ModeConsumption]]:
        return self.task_mode_dict

    def _get_successors(self) -> Dict[int, List[int]]:
        return self.successors

    def _get_tasks_ids(self) -> Union[Set[int], Dict[int, Any], List[int]]:
        return self.task_ids

    def _get_task_duration(self, task: int, mode: Optional[int] = 1, progress_from: Optional[float] = 0.) -> int:
        return self.duration_dict[task][mode]

    def _get_original_quantity_resource(self, resource: str, **kwargs) -> int:
        return self.resource_availability[resource]

    def _get_resource_renewability(self) -> Dict[str, bool]:
        return self.resource_renewable

    def _get_all_resources_skills(self) -> Dict[str, Dict[str, Any]]:
        return self.resource_skills

    def _get_all_tasks_skills(self) -> Dict[int, Dict[int, Dict[str, Any]]]:
        return self.task_skills_dict

    def _get_objectives(self) -> List[int]:
        return [SchedulingObjectiveEnum.MAKESPAN]


class D(MultiModeMultiSkillRCPSPCalendar):
    pass


class MSRCPSPCalendar(D):
    def _get_max_horizon(self) -> int:
        return self.max_horizon

    def _get_quantity_resource(self, resource: str, time: int, **kwargs) -> int:
        return self.resource_availability[resource][time]

    def __init__(self,
                 skills_names: List[str] = None,
                 resource_unit_names: List[str] = None,
                 resource_type_names: List[str] = None,
                 resource_skills: Dict[str, Dict[str, Any]]=None,
                 task_ids: List[int] = None,
                 tasks_mode: Dict[int, Dict[int, Dict[str, int]]] = None,
                 successors: Dict[int, List[int]] = None,
                 max_horizon: int = None,
                 resource_availability: Dict[str, List[int]] = None,
                 resource_renewable: Dict[str, bool] = None):
        self.skills_set = set(skills_names)
        self.resource_unit_names = resource_unit_names
        self.resource_type_names = resource_type_names
        self.resource_skills = resource_skills
        self.task_ids = task_ids
        self.tasks_mode = tasks_mode
        # transform the "mode_details" dict that we largely used in DO in the good format.
        self.task_mode_dict = {}
        self.task_skills_dict = {}
        self.duration_dict = {}
        for task in self.tasks_mode:
            self.task_mode_dict[task] = {}
            self.task_skills_dict[task] = {}
            self.duration_dict[task] = {}
            for mode in self.tasks_mode[task]:
                self.task_mode_dict[task][mode] = ConstantModeConsumption({})
                self.task_skills_dict[task][mode] = {}
                for r in self.tasks_mode[task][mode]:
                    if r in self.resource_type_names:
                        self.task_mode_dict[task][mode].mode_details[r] = [self.tasks_mode[task][mode][r]]
                    if r in self.skills_set:
                        self.task_skills_dict[task][mode][r] = self.tasks_mode[task][mode][r]
                self.duration_dict[task][mode] = self.tasks_mode[task][mode]["duration"]
        self.successors = successors
        self.max_horizon = max_horizon
        self.resource_availability = resource_availability
        self.resource_renewable = resource_renewable
        self.initialize_domain()

    def _get_resource_units_names(self) -> List[str]:
        """Return the names (string) of all resource units as a list."""
        return self.resource_unit_names

    def _get_resource_types_names(self) -> List[str]:
        return self.resource_type_names

    def _get_resource_type_for_unit(self) -> Dict[str, str]:
        """Return a dictionary where the key is a resource unit name and the value a resource type name.
        An empty dictionary can be used if there are no resource unit matching a resource type."""
        return None

    def get_max_horizon(self) -> int:
        return self.max_horizon

    def _get_tasks_modes(self) -> Dict[int, Dict[int, ModeConsumption]]:
        return self.task_mode_dict

    def _get_successors(self) -> Dict[int, List[int]]:
        return self.successors

    def _get_tasks_ids(self) -> Union[Set[int], Dict[int, Any], List[int]]:
        return self.task_ids

    def _get_task_duration(self, task: int, mode: Optional[int] = 1, progress_from: Optional[float] = 0.) -> int:
        return self.duration_dict[task][mode]

    def _get_original_quantity_resource(self, resource: str, **kwargs) -> int:
        return self.resource_availability[resource]

    def _get_resource_renewability(self) -> Dict[str, bool]:
        return self.resource_renewable

    def _get_all_resources_skills(self) -> Dict[str, Dict[str, Any]]:
        return self.resource_skills

    def _get_all_tasks_skills(self) -> Dict[int, Dict[int, Dict[str, Any]]]:
        return self.task_skills_dict

    def _get_objectives(self) -> List[int]:
        return [SchedulingObjectiveEnum.MAKESPAN]


if __name__ == "__main__":
    from skdecide.hub.domain.rcpsp.rcpsp_sk_parser import load_domain
    domain = load_domain()
    state = domain.get_initial_state()
    print("Initial state : ", state)
    actions = domain.get_applicable_actions(state)
    print([str(action) for action in actions.get_elements()])
    action = actions.get_elements()[0]
    new_state = domain.get_next_state(state, action)
    print("New state ", new_state)
    actions = domain.get_applicable_actions(new_state)
    print("New actions : ", [str(action) for action in actions.get_elements()])
    action = actions.get_elements()[0]
    print(action)
    new_state = domain.get_next_state(new_state, action)
    print("New state :", new_state)
    print('_is_terminal: ', domain._is_terminal(state))


