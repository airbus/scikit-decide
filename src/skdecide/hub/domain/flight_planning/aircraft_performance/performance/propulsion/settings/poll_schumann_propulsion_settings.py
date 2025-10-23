from dataclasses import dataclass

from skdecide.hub.domain.flight_planning.aircraft_performance.performance.performance_model_enum import (
    PerformanceModelEnum,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.performance.propulsion.settings.propulsion_settings_interface import (
    PropulsionSettings,
)


@dataclass
class PollSchumannPropulsionSettings(PropulsionSettings):
    """
    Propulsion settings for OpenAP models
    """

    ac_parameters: dict[str, float] = None
    sref: float = None

    def __post_init__(self):
        self.performance_model_type = PerformanceModelEnum.POLL_SCHUMANN
