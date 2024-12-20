from copy import deepcopy

from skdecide.hub.domain.flight_planning.aircraft_performance.bean.aircraft_state import (
    AircraftState,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.performance.performance_model_enum import (
    PerformanceModelEnum,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.performance.phase_enum import (
    PhaseEnum,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.performance.propulsion.service._openap_propulsion_service import (
    _OpenapPropulsionService,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.performance.propulsion.service._poll_schumann_propulsion_service import (
    _PollSchumannPropulsionService,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.performance.propulsion.settings.propulsion_settings_interface import (
    PropulsionSettings,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.performance.rating_enum import (
    RatingEnum,
)


class PropulsionService:
    """
    Main propulsion service class to be handled
    """

    def __init__(self):
        self.all_propulsion_services = {
            PerformanceModelEnum.OPENAP: _OpenapPropulsionService(),
            PerformanceModelEnum.POLL_SCHUMANN: _PollSchumannPropulsionService(),
        }

    def init_settings(
        self, model_path: str, performance_model_type: PerformanceModelEnum
    ) -> PropulsionSettings:
        return self.all_propulsion_services[performance_model_type].init_settings(
            model_path
        )

    def compute_total_net_thrust_n(
        self,
        propulsion_settings: PropulsionSettings,
        aircraft_state: AircraftState,
    ) -> float:
        return self.all_propulsion_services[
            propulsion_settings.performance_model_type
        ].compute_total_net_thrust_n(propulsion_settings, aircraft_state)

    def compute_total_fuel_flow_kg_per_sec(
        self,
        propulsion_settings: PropulsionSettings,
        aircraft_state: AircraftState,
    ) -> float:
        return self.all_propulsion_services[
            propulsion_settings.performance_model_type
        ].compute_total_fuel_flow_kg_per_sec(propulsion_settings, aircraft_state)

    def compute_min_rating(
        self,
        propulsion_settings: PropulsionSettings,
        aircraft_state: AircraftState,
    ) -> RatingEnum:
        return self.all_propulsion_services[
            propulsion_settings.performance_model_type
        ].compute_min_rating(propulsion_settings, aircraft_state)

    def compute_max_rating(
        self,
        propulsion_settings: PropulsionSettings,
        aircraft_state: AircraftState,
    ) -> RatingEnum:
        return self.all_propulsion_services[
            propulsion_settings.performance_model_type
        ].compute_max_rating(propulsion_settings, aircraft_state)

    def compute_tsp_from_thrust(
        self,
        propulsion_settings: PropulsionSettings,
        aircraft_state: AircraftState,
        target_thrust_n: float,
    ) -> float:
        aircraft_state_copy = deepcopy(aircraft_state)
        if aircraft_state.is_one_eo:
            aircraft_state_copy.rating_level = RatingEnum.MCN
        elif aircraft_state.phase == PhaseEnum.TAKE_OFF:
            aircraft_state_copy.rating_level = RatingEnum.T0N
        else:
            aircraft_state_copy.rating_level = RatingEnum.MCL

        tsp_max = self.compute_max_rating(
            propulsion_settings=propulsion_settings, aircraft_state=aircraft_state_copy
        )
        aircraft_state_copy.tsp = tsp_max
        thrust_max = self.compute_total_net_thrust_n(
            propulsion_settings=propulsion_settings, aircraft_state=aircraft_state_copy
        )

        if target_thrust_n > thrust_max:
            raise Exception("Target thrust exceeds min")

        tsp_min = self.compute_min_rating(
            propulsion_settings=propulsion_settings, aircraft_state=aircraft_state_copy
        )
        aircraft_state_copy.tsp = tsp_min
        thrust_min = self.compute_total_net_thrust_n(
            propulsion_settings=propulsion_settings, aircraft_state=aircraft_state_copy
        )

        if target_thrust_n < thrust_min:
            raise Exception("Target thrust exceeds min")

        computed_thrust_n = 0
        trial_tsp = tsp_min + (tsp_max - tsp_min) / (thrust_max - thrust_min) * (
            target_thrust_n - thrust_min
        )
        while abs(target_thrust_n - computed_thrust_n) > 0.5:
            aircraft_state_copy.tsp = trial_tsp
            computed_thrust_n = self.compute_total_net_thrust_n(
                propulsion_settings=propulsion_settings,
                aircraft_state=aircraft_state_copy,
            )
            if thrust_min < computed_thrust_n < target_thrust_n:
                tsp_min = trial_tsp
                thrust_min = computed_thrust_n
            elif thrust_max > computed_thrust_n > target_thrust_n:
                tsp_max = trial_tsp
                thrust_max = computed_thrust_n

            trial_tsp = tsp_min + (tsp_max - tsp_min) / (thrust_max - thrust_min) * (
                target_thrust_n - thrust_min
            )
        return trial_tsp
