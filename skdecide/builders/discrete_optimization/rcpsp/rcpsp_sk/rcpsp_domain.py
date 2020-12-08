from enum import Enum
from typing import *

from skdecide import *
from skdecide.builders.domain import *
from skdecide.builders.discrete_optimization.rcpsp.rcpsp_model import RCPSPModel
from heapq import heappop, heappush


# Example of State type (adapt to your needs)
class StateRcpsp:
    current_time: float
    current_ressource_capacity: Union[List[float],
                                      Dict[str, float]]
    current_activity_queue: List[Dict[str, int]]
    past_activity: Set[int]

    def __init__(self, current_time: float,
                 current_ressource_capacity: Union[List[float], Dict[str, float]],
                 current_activity_queue: List[Dict[int, int]],
                 past_activity: Set[int]):
        self.current_time = current_time
        self.current_ressource_capacity = current_ressource_capacity
        self.current_activity_queue = current_activity_queue
        self.past_activity = past_activity


class EventEnum(Enum):
    START = 0
    END = 1


# Example of Action type (adapt to your needs)
class Action(Enum):
    up = 0
    down = 1
    left = 2
    right = 3


class D(Domain,
        SingleAgent,
        Sequential,
        EnumerableTransitions,
        Goals,
        DeterministicInitialized,
        Markovian,
        FullyObservable,
        PositiveCosts):
    T_state = StateRcpsp  # Type of states
    T_observation = T_state  # Type of observations
    T_event = Action  # Type of events
    T_value = float  # Type of transition values (rewards or costs)
    T_info = None  # Type of additional information in environment outcome


class RcpspDomain(D):

    def _get_transition_value(self, memory: D.T_state, action: D.T_event,
                              next_state: Optional[D.T_state] = None) -> TransitionValue[D.T_value]:
        pass

    def _get_next_state_distribution(self, memory: D.T_state, action: D.T_event) -> SingleValueDistribution[D.T_state]:
        pass

    def _is_terminal(self, state: D.T_state) -> bool:
        pass

    def _get_applicable_actions_from(self, memory: D.T_state) -> Space[D.T_event]:
        pass

    def _get_action_space_(self) -> Space[D.T_event]:
        pass

    def _get_enabled_events_from(self, memory: D.T_state) -> Space[D.T_event]:
        pass

    def _get_goals_(self) -> Space[D.T_observation]:
        pass

    def _get_initial_state_(self) -> D.T_state:
        pass

    def _get_observation_space_(self) -> Space[D.T_observation]:
        pass

