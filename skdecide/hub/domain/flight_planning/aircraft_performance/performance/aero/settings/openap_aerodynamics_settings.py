from dataclasses import dataclass

from openap.drag import Drag

from skdecide.hub.domain.flight_planning.aircraft_performance.performance.aero.settings.aerodynamics_settings_interface import (
    AerodynamicsSettings,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.performance.performance_model_enum import (
    PerformanceModelEnum,
)


@dataclass
class OpenapAerodynamicsSettings(AerodynamicsSettings):
    """
    Aerodynamics settings for OpenAP models
    """

    drag: Drag = None
    sref: float = None

    def __post_init__(self):
        self.performance_model_type = PerformanceModelEnum.OPENAP
