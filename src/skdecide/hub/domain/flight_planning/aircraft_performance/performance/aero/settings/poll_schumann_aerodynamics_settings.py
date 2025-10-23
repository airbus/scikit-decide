from dataclasses import dataclass

from skdecide.hub.domain.flight_planning.aircraft_performance.performance.aero.settings.aerodynamics_settings_interface import (
    AerodynamicsSettings,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.performance.performance_model_enum import (
    PerformanceModelEnum,
)


@dataclass
class PollSchumannAerodynamicsSettings(AerodynamicsSettings):
    """
    Aerodynamics settings for Poll Schumann models
    """

    ac_parameters: dict[str, float] = None
    sref: float = None

    def __post_init__(self):
        self.performance_model_type = PerformanceModelEnum.POLL_SCHUMANN
