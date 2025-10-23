from abc import ABC
from dataclasses import dataclass

from skdecide.hub.domain.flight_planning.aircraft_performance.performance.performance_model_enum import (
    PerformanceModelEnum,
)


@dataclass
class PropulsionSettings(ABC):
    """
    Abstract dataclass for propulsion service settings
    """

    performance_model_type: PerformanceModelEnum = None
    nb_engines: int = None
    sref: float = None
