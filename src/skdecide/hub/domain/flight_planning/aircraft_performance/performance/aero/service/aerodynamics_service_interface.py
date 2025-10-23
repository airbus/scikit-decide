from abc import ABC, abstractmethod

from skdecide.hub.domain.flight_planning.aircraft_performance.bean.aircraft_state import (
    AircraftState,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.performance.aero.settings.aerodynamics_settings_interface import (
    AerodynamicsSettings,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.performance.performance_model_enum import (
    PerformanceModelEnum,
)


class AerodynamicsServiceInterface(ABC):
    """
    Interface for Aerodynamics Services
    """

    @property
    @abstractmethod
    def performance_model_type(self) -> PerformanceModelEnum:
        raise NotImplementedError

    @abstractmethod
    def init_settings(
        self, model_path: str, performance_model_type: PerformanceModelEnum
    ) -> AerodynamicsSettings:
        raise NotImplementedError

    @abstractmethod
    def compute_drag_coefficient(
        self, settings: AerodynamicsSettings, aircraft_state: AircraftState
    ) -> float:
        raise NotImplementedError

    @abstractmethod
    def compute_crossover(
        self, settings: AerodynamicsSettings, aircraft_state: AircraftState
    ) -> float:
        raise NotImplementedError
