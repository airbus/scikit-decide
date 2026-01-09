from skdecide.hub.domain.flight_planning.aircraft_performance.bean.aircraft_state import (
    AircraftState,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.performance.performance_model_enum import (
    PerformanceModelEnum,
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
        aircraft_state_copy = aircraft_state.clone()

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

        # Handle edge cases where target_thrust is very close to min/max
        if abs(target_thrust_n - thrust_min) < 0.5:
            return tsp_min

        if abs(target_thrust_n - thrust_max) < 0.5:
            return tsp_max

        # Initial guess for trial_tsp using linear interpolation (secant method's first step)
        # Ensure thrust_max - thrust_min is not zero before division
        if (
            abs(thrust_max - thrust_min) < 1e-9
        ):  # very small difference, close enough to either bound
            return tsp_min  # or tsp_max, doesn't matter much

        trial_tsp = tsp_min + (tsp_max - tsp_min) / (thrust_max - thrust_min) * (
            target_thrust_n - thrust_min
        )

        computed_thrust_n = 0
        iteration_count = 0
        max_iterations = 100  # Add a safety break for infinite loops
        while abs(target_thrust_n - computed_thrust_n) > 10:
            iteration_count += 1
            if iteration_count > max_iterations:
                print(
                    f"Warning: Max iterations ({max_iterations}) reached. Returning current trial_tsp."
                )
                return trial_tsp  # Or raise an exception, depending on desired behavior

            # Ensure trial_tsp stays within valid TSP range
            # This is crucial if the interpolation formula leads to values outside [tsp_min_initial, tsp_max_initial]
            trial_tsp = max(tsp_min, min(tsp_max, trial_tsp))
            aircraft_state_copy.tsp = trial_tsp
            computed_thrust_n = self.compute_total_net_thrust_n(
                propulsion_settings=propulsion_settings,
                aircraft_state=aircraft_state_copy,
            )

            # Update bounds based on where computed_thrust_n falls relative to target_thrust_n
            if computed_thrust_n < target_thrust_n:
                tsp_min = trial_tsp
                thrust_min = computed_thrust_n
            else:  # computed_thrust_n >= target_thrust_n
                tsp_max = trial_tsp
                thrust_max = computed_thrust_n

            # Re-calculate trial_tsp for the next iteration
            # Add a safeguard for division by zero or very small numbers
            thrust_range = thrust_max - thrust_min

            if (
                abs(thrust_range) < 1e-9
            ):  # If the thrust range becomes too small, we've likely converged or are stuck
                print(
                    f"Warning: Thrust range ({thrust_range}) too small. Returning current trial_tsp."
                )
                break  # Exit loop

            trial_tsp = tsp_min + (tsp_max - tsp_min) / thrust_range * (
                target_thrust_n - thrust_min
            )

        return trial_tsp
