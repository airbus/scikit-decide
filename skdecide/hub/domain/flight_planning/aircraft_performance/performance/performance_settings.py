from dataclasses import dataclass

from skdecide.hub.domain.flight_planning.aircraft_performance.performance.aero.settings.aerodynamics_settings_interface import (
    AerodynamicsSettings,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.performance.propulsion.settings.propulsion_settings_interface import (
    PropulsionSettings,
)


@dataclass
class PerformanceSettings:
    """
    Dataclass to aggregate all the performance settings
    """

    aerodynamics_settings: AerodynamicsSettings
    propulsion_settings: PropulsionSettings
