from skdecide.hub.domain.flight_planning.aircraft_performance.bean.aircraft_state import (
    AircraftState,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.performance.aero.service._openap_aerodynamics_service import (
    _OpenapAerodynamicsService,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.performance.aero.service._poll_schumann_aerodynamics_service import (
    _PollSchumannAerodynamicsService,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.performance.aero.settings.aerodynamics_settings_interface import (
    AerodynamicsSettings,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.performance.performance_model_enum import (
    PerformanceModelEnum,
)


class AerodynamicsService:
    """
    Main aerodynamics service class to be handled
    """

    def __init__(self):
        self.all_aerodynamics_services = {
            PerformanceModelEnum.OPENAP: _OpenapAerodynamicsService(),
            PerformanceModelEnum.POLL_SCHUMANN: _PollSchumannAerodynamicsService(),
        }

    def init_settings(
        self, model_path: str, performance_model_type: PerformanceModelEnum
    ) -> AerodynamicsSettings:
        return self.all_aerodynamics_services[performance_model_type].init_settings(
            model_path
        )

    def compute_drag_coefficient(
        self, aerodynamics_settings: AerodynamicsSettings, aircraft_state: AircraftState
    ) -> float:

        return self.all_aerodynamics_services[
            aerodynamics_settings.performance_model_type
        ].compute_drag_coefficient(aerodynamics_settings, aircraft_state)

    def compute_crossover(
        self, aerodynamics_settings: AerodynamicsSettings, aircraft_state: AircraftState
    ) -> float:

        return self.all_aerodynamics_services[
            aerodynamics_settings.performance_model_type
        ].compute_crossover(aerodynamics_settings, aircraft_state)
