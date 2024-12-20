from abc import ABC, abstractmethod

from skdecide.hub.domain.flight_planning.aircraft_performance.bean.aircraft_state import (
    AircraftState,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.performance.performance_model_enum import (
    PerformanceModelEnum,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.performance.propulsion.settings.propulsion_settings_interface import (
    PropulsionSettings,
)


class PropulsionServiceInterface(ABC):
    """
    Interface defining propulsion services
    """

    @property
    @abstractmethod
    def performance_model_type(self) -> PerformanceModelEnum:
        raise NotImplementedError

    @abstractmethod
    def init_settings(
        self, model_path: str, performance_model_type: PerformanceModelEnum
    ) -> PropulsionSettings:
        raise NotImplementedError

    @abstractmethod
    def compute_total_net_thrust_n(
        self, propulsion_settings: PropulsionSettings, aircraft_state: AircraftState
    ) -> float:
        raise NotImplementedError

    @abstractmethod
    def compute_total_fuel_flow_kg_per_sec(
        self, propulsion_settings: PropulsionSettings, aircraft_state: AircraftState
    ) -> float:
        raise NotImplementedError

    @abstractmethod
    def compute_max_rating(
        self, propulsion_settings: PropulsionSettings, aircraft_state: AircraftState
    ) -> float:
        raise NotImplementedError

    @abstractmethod
    def compute_min_rating(
        self, propulsion_settings: PropulsionSettings, aircraft_state: AircraftState
    ) -> float:
        raise NotImplementedError
