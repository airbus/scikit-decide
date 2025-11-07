from abc import ABC
from dataclasses import dataclass

from skdecide.hub.domain.flight_planning.aircraft_performance.performance.performance_model_enum import (
    PerformanceModelEnum,
)


@dataclass
class AerodynamicsSettings(ABC):
    """
    Dataclass to aggregate all the aerodynamics settings
    """

    performance_model_type: PerformanceModelEnum = None
    sref: float = None
