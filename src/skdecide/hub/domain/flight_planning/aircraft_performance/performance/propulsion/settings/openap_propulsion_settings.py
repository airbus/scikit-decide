from dataclasses import dataclass

from openap.fuel import FuelFlow
from openap.thrust import Thrust

from skdecide.hub.domain.flight_planning.aircraft_performance.performance.performance_model_enum import (
    PerformanceModelEnum,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.performance.propulsion.settings.propulsion_settings_interface import (
    PropulsionSettings,
)


@dataclass
class OpenapPropulsionSettings(PropulsionSettings):
    """
    Propulsion settings for OpenAP models
    """

    fuel: FuelFlow = None
    thrust: Thrust = None

    def __post_init__(self):
        self.performance_model_type = PerformanceModelEnum.OPENAP
