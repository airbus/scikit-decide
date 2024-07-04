# data and maths
import math

# typing
from typing import Any, Dict, Optional

import numpy as np

# engine data loader
from skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.engine_loader import (
    load_aircraft_engine_params,
)

# utils
from skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.parameters import (
    atmospheric_parameters as atm_params,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.parameters import (
    jet,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.parameters import (
    operational_limits as op_lim,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.parameters import (
    units,
)

# aero
from skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.utils.aero import (
    ft,
    kts,
)

MIN_CRUISE_ALTITUDE = 20_000.0


class FuelFlow:
    """
    Fuel flow model based on Poll-Schumann model.
    """

    def __init__(self, actype: str):
        """Initialisation of the fuel flow model based on Poll-Schumann model.

        # Parameters
            actype (str):
                Aircraft type for which the fuel flow model is created.
        """

        # load engine parameters for selected aircraft
        self.aircraft_parameters = load_aircraft_engine_params(actype)

    def __call__(
        self,
        values_current: Dict[str, float],
        delta_time: float,
        vs: Optional[float] = 0.0,
    ) -> float:
        """Compute fuel flow based on Poll-Schumann model.

        # Parameters
            values_current (Dict[str, float]):
                Dictionary with current values of altitude [:math:`ft`], speed [:math:`kts`], temperature [:math:`K`], and mass [:math:`Kg`].
            delta_time (float):
                Time step in seconds [:math:`s`].
            vs (Optional[float], optional):
                Vertical speed [:math: `ft/min`]. Defaults to 0.0 ft/min.

        # Returns
            float: Fuel flow, [:math:`Kg/s`].
        """

        # values current
        altitude_current = values_current["alt"]
        air_temperature = values_current["temp"]
        speed_current = values_current["speed"] * kts
        mass_current = values_current["mass"]

        # values next
        altitude_next = altitude_current + vs * delta_time / 60  # s -> min
        # atmospheric quantities
        air_pressure = units.ft_to_pl(altitude_current) * 100.0

        # clip unrealistically high TAS
        max_mach = op_lim.max_mach_number_by_altitude(
            altitude_current,
            air_pressure,
            self.aircraft_parameters["max_mach_num"],
            self.aircraft_parameters["p_i_max"],
            self.aircraft_parameters["p_inf_co"],
            atm_speed_limit=False,
            buffer=0.02,
        )

        true_airspeed, mach_num = jet.clip_mach_number(
            speed_current, air_temperature, max_mach
        )

        # Reynolds number
        reynolds = atm_params.reynolds_number(
            self.aircraft_parameters["wing_span"],
            self.aircraft_parameters["wing_aspect_ratio"],
            mach_num,
            air_temperature,
            air_pressure,
        )

        # Rate Of Climb and Descent; path angle; acceleration
        rocd = jet.rate_of_climb_descent(altitude_current, altitude_next, delta_time)
        path_angle = math.degrees(
            (altitude_next - altitude_current) * ft / (true_airspeed * delta_time)
        )  # approximation for small angles

        dv_dt = jet.acceleration(true_airspeed, delta_time)

        # aircraft performance parameters
        c_lift = atm_params.lift_coefficient(
            self.aircraft_parameters["wing_surface_area"],
            mass_current,
            air_pressure,
            air_temperature,
            mach_num,
            path_angle,
        )
        c_f = atm_params.skin_friction_coefficient(reynolds)
        c_drag_0 = atm_params.zero_lift_drag_coefficient(
            c_f, self.aircraft_parameters["psi_0"]
        )
        e_ls = atm_params.oswald_efficiency_factor(c_drag_0, self.aircraft_parameters)
        c_drag_w = atm_params.wave_drag_coefficient(
            mach_num, c_lift, self.aircraft_parameters
        )
        c_drag = atm_params.airframe_drag_coefficient(
            c_drag_0,
            c_drag_w,
            c_lift,
            e_ls,
            self.aircraft_parameters["wing_aspect_ratio"],
        )

        # engine parameters and fuel consumption
        thrust = atm_params.thrust_force(
            mass_current, c_lift, c_drag, dv_dt, path_angle
        )

        c_t = atm_params.engine_thrust_coefficient(
            thrust,
            mach_num,
            air_pressure,
            self.aircraft_parameters["wing_surface_area"],
        )
        c_t_eta_b = atm_params.thrust_coefficient_at_max_efficiency(
            mach_num,
            self.aircraft_parameters["m_des"],
            self.aircraft_parameters["c_t_des"],
        )

        # correct fuel flow
        c_t_available = op_lim.max_available_thrust_coefficient(
            air_temperature, mach_num, c_t_eta_b, self.aircraft_parameters
        )
        c_t = np.clip(c_t, 0, c_t_available)

        # engine efficiency
        eta_over_eta_b_min = 0.5
        engine_efficiency = atm_params.overall_propulsion_efficiency(
            mach_num, c_t, c_t_eta_b, self.aircraft_parameters, eta_over_eta_b_min
        )

        # fuel flow
        q_fuel = 43.13e6
        fuel_flow = atm_params.fuel_mass_flow_rate(
            air_pressure,
            air_temperature,
            mach_num,
            c_t,
            engine_efficiency,
            self.aircraft_parameters["wing_surface_area"],
            q_fuel,
        )

        # compute flight phase
        rocd_threshold = 250.0
        cruise = (
            (rocd < rocd_threshold)
            & (rocd > -rocd_threshold)
            & (altitude_current > MIN_CRUISE_ALTITUDE)
        )
        climb = ~cruise & (rocd > 0.0)
        # descent = ~cruise & (rocd < 0.0)

        # convert to string
        flight_phase = np.where(cruise, "cruise", np.where(climb, "climb", "descent"))

        self.fuel_flow = atm_params.fuel_flow_correction(
            fuel_flow,
            altitude_current,
            air_temperature,
            air_pressure,
            mach_num,
            self.aircraft_parameters["ff_idle_sls"],
            self.aircraft_parameters["ff_max_sls"],
            flight_phase,
        )

        return self.fuel_flow
