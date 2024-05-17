# math and data
import math

# typing
from typing import Dict

import numpy as np

from skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.parameters import (
    constants,
    jet,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.parameters import (
    operational_limits as op_lim,
)

# ----------------------
# Atmospheric parameters
# ----------------------


def reynolds_number(
    wing_span: float,
    wing_aspect_ratio: float,
    mach_num: float,
    air_temperature: float,
    air_pressure: float,
) -> float:
    """Calculate the Reynolds number.

    # Parameters
        wing_span (float):
            Wing surface area, [:math:`m`]
        wing_aspect_ratio (float):
            Wing aspect ratio, [:math:`-`]
        mach_num (float):
            Mach number, [:math:`-`]
        air_temperature (float):
            Air temperature, [:math:`K`]
        air_pressure (float):
            Air pressure, [:math:`Pa`]

    # Returns
        float: Reynolds number, [:math:`-`]
    """
    # compute wing chord
    wing_chord = wing_span / wing_aspect_ratio

    # dynamic viscosity
    mu_inf = dynamic_viscosity(air_temperature)

    # mach number
    M_inf = mach_num

    # local speed of sound
    a_inf = local_speed_of_sound(air_temperature)

    # density of dry air
    rho_inf = air_density(air_pressure, air_temperature)

    return (rho_inf * a_inf / mu_inf) * wing_chord * M_inf


def local_speed_of_sound(air_temperature: float) -> float:
    """Calculate the local speed of sound.

    # Parameters
        air_temperature (float):
            Air temperature, [:math:`K`]

    # Returns
        float: Local speed of sound, [:math:`m/s`]
    """
    return (constants.gamma * constants.R * air_temperature) ** 0.5


def air_density(air_pressure: float, air_temperature: float) -> float:
    """Calculate the air density.

    # Parameters
        air_pressure (float):
            Air pressure, [:math:`Pa`]
        air_temperature (float):
            Air temperature, [:math:`K`]

    # Returns
        float: Air density, [:math:`kg/m^3`]
    """
    return air_pressure / (constants.R * (air_temperature + 1e-8))


def dynamic_viscosity(air_temperature: float) -> float:
    """Calculate approximation of the dynamic viscosity.

    # Parameters
        air_temperature (float):
            Air temperature, [:math:`K`]

    # Returns
        float: Dynamic viscosity, [:math:`kg m^{-1} s^{-1}`]
    """

    mu_Tref = 1.715e-5
    Tref = 273.15
    S = 110.4

    return (
        mu_Tref
        * (air_temperature / Tref) ** 1.5
        * (Tref + S)
        / (air_temperature + S + 1e-6)
    )


# -------------------------------
# Lift and drag coefficients
# -------------------------------


def lift_coefficient(
    wing_surface_area: float,
    aircraft_mass: float,
    air_pressure: float,
    air_temperature: float,
    mach_num: float,
    climb_angle: float,
) -> float:
    """Calculate the lift coefficient.

    # Parameters
        wing_surface_area (float):
            Wing surface area, [:math:`m^2`]
        aircraft_mass (float):
            Aircraft mass, [:math:`kg`]
        air_pressure (float):
            Air pressure, [:math:`Pa`]
        mach_num (float):
            Mach number, [:math:`-`]
        climb_angle (float):
            Climb angle, [:math:`deg`]

    # Returns
        float: Lift coefficient, [:math:`-`]
    """

    lift_force = aircraft_mass * constants.g * np.cos(np.deg2rad(climb_angle))
    air_speed = mach_num * local_speed_of_sound(air_temperature)
    air_dens = air_density(air_pressure, air_temperature)
    dynamic_pressure = air_dens * air_speed**2 / 2

    return lift_force / (dynamic_pressure * wing_surface_area)


def skin_friction_coefficient(reynolds_number: float) -> float:
    """Calculate the skin friction coefficient.

    # Parameters
        reynolds_number (float):
            Reynolds number, [:math:`-`]

    # Returns
        float: Skin friction coefficient, [:math:`-`]
    """
    return 0.0269 / (reynolds_number**0.14)


def zero_lift_drag_coefficient(c_f: float, psi_0: float) -> float:
    """Calculate the zero-lift drag coefficient.

    # Parameters
        c_f (float):
            Skin friction coefficient, [:math:`-`]
        psi_0 (float):
            Miscellaneous drag factor, [:math:`-`]

    # Returns
        float: Zero-lift drag coefficient, [:math:`-`]
    """
    return c_f * psi_0


def oswald_efficiency_factor(
    c_drag_0: float, aircraft_parameters: Dict[str, float]
) -> float:
    """Calculate the Oswald efficiency factor.

    # Parameters
        c_drag_0 (float):
            Zero-lift drag coefficient, [:math:`-`]
        aircraft_parameters (Dict[str, float]):
            Aircraft parameters.

    # Returns
        float: Oswald efficiency factor, [:math:`-`]
    """
    numer = 1.075 if aircraft_parameters["winglets"] == "no" else 1.0
    k1 = _non_vortex_lift_dependent_drag_factor(
        c_drag_0, aircraft_parameters["cos_sweep"]
    )
    denom = 1.04 + math.pi * k1 * aircraft_parameters["wing_aspect_ratio"]

    return numer / denom


def _non_vortex_lift_dependent_drag_factor(c_drag_0: float, cos_sweep: float) -> float:
    """Calculate the miscellaneous lift-dependent drag factor.

    # Parameters
        c_drag_0 (float):
            Zero-lift drag coefficient, [:math:`-`]
        cos_sweep (float):
            Cosine of the wing sweep angle, [:math:`-`]

    # Returns
        float: Miscellaneous lift-dependent drag factor, [:math:`-`]
    """
    return 0.80 * (1 - 0.53 * cos_sweep) * c_drag_0


def wave_drag_coefficient(
    mach_num: float, c_lift: float, aircraft_parameters: Dict[str, float]
) -> float:
    """Calculate the wave drag coefficient.

    # Parameters
        mach_num (float):
            Mach number, [:math:`-`]
        c_lift (float):
            Lift coefficient, [:math:`-`]
        aircraft_parameters (Dict[str, float]):
            Aircraft parameters.

    # Returns
        float: Wave drag coefficient, [:math:`-`]
    """
    m_cc = aircraft_parameters["wing_constant"] - 0.10 * (
        c_lift / aircraft_parameters["cos_sweep"] ** 2
    )
    x = mach_num * aircraft_parameters["cos_sweep"] / m_cc

    c_d_w = np.where(
        x < aircraft_parameters["j_2"],
        0.0,
        aircraft_parameters["cos_sweep"] ** 3
        * aircraft_parameters["j_1"]
        * (x - aircraft_parameters["j_2"]) ** 2,
    )

    output = np.where(
        x < aircraft_parameters["x_ref"],
        c_d_w,
        c_d_w + aircraft_parameters["j_3"] * (x - aircraft_parameters["x_ref"]) ** 4,
    )

    return output


def airframe_drag_coefficient(
    c_drag_0: float,
    c_drag_w: float,
    c_lift: float,
    e_ls: float,
    wing_aspect_ratio: float,
) -> float:
    """Calculate total airframe drag coefficient.

    # Parameters
        c_drag_0 (float):
            Zero-lift drag coefficient, [:math:`-`]
        c_drag_w (float):
            Wave drag coefficient, [:math:`-`]
        c_lift (float):
            Lift coefficient, [:math:`-`]
        e_ls (float):
            Oswald efficiency factor, [:math:`-`]
        wing_aspect_ratio (float):
            Wing aspect ratio, [:math:`-`]

    # Returns
        float: Total airframe drag coefficient, [:math:`-`]
    """
    return c_drag_0 + c_drag_w + c_lift**2 / (math.pi * e_ls * wing_aspect_ratio)


# -------------------
# Engine parameters
# -------------------


def thrust_force(
    aircraft_mass: float, c_l: float, c_d: float, dv_dt: float, theta: float
) -> float:
    """Calculate thrust force summed over all engines.

    # Parameters
        aircraft_mass (float):
            Aircraft mass, [:math:`kg`]
        c_l (float):
            Lift coefficient, [:math:`-`]
        c_d (float):
            Drag coefficient, [:math:`-`]
        dv_dt (float):
            Rate of change of velocity, [:math:`m/s^2`]
        theta (float):
            Flight path angle, [:math:`deg`]

    # Returns
        float: Thrust force, [:math:`N`]
    """
    theta_rad = math.radians(theta)

    f_thrust = (
        (aircraft_mass * constants.g * math.cos(theta_rad) * (c_d / c_l))
        + (aircraft_mass * constants.g * math.sin(theta_rad))
        + (aircraft_mass * dv_dt)
    )
    return max(f_thrust, 0)


def engine_thrust_coefficient(
    f_thrust: float, mach_num: float, air_pressure: float, wing_surface_area: float
) -> float:
    """Calculate engine thrust coefficient.

    # Parameters
        f_thrust (float):
            Thrust force, [:math:`N`]
        mach_num (float):
            Mach number, [:math:`-`]
        air_pressure (float):
            Air pressure, [:math:`Pa`]
        wing_surface_area (float):
            Wing surface area, [:math:`m^2`]

    # Returns
        float: Engine thrust coefficient, [:math:`-`]
    """
    return f_thrust / (
        0.5 * constants.gamma * air_pressure * mach_num**2 * wing_surface_area
    )


def overall_propulsion_efficiency(
    mach_num: float,
    c_t: float,
    c_t_eta_b: float,
    aircraft_parameters: Dict[str, float],
    eta_over_eta_b_min: float,
) -> float:
    """Calculate overall propulsion efficiency.

    # Parameters
        mach_num (float):
            Mach number, [:math:`-`]
        c_t (float):
            Thrust coefficient, [:math:`-`]
        c_t_eta_b (float):
            Thrust coefficient at maximum efficiency, [:math:`-`]
        aircraft_parameters (Dict[str, float]):
            Aircraft parameters.
        eta_over_eta_b_min (float):
            Minimum engine efficiency, [:math:`-`]

    # Returns
        float: Overall propulsion efficiency, [:math:`-`]
    """
    eta_over_eta_b = propulsion_efficiency_over_max_propulsion_efficiency(
        mach_num, c_t, c_t_eta_b
    )

    if eta_over_eta_b_min is not None:
        eta_over_eta_b.clip(min=eta_over_eta_b_min, out=eta_over_eta_b)

    eta_b = max_overall_propulsion_efficiency(
        mach_num, aircraft_parameters["eta_1"], aircraft_parameters["eta_2"]
    )

    return eta_over_eta_b * eta_b


def propulsion_efficiency_over_max_propulsion_efficiency(
    mach_num: float, c_t: float, c_t_eta_b: float
) -> float:
    """Calculate propulsion efficiency over maximum propulsion efficiency.

    # Parameters
        mach_num (float):
            Mach number, [:math:`-`]
        c_t (float):
            Thrust coefficient, [:math:`-`]
        c_t_eta_b (float):
            Thrust coefficient at maximum efficiency, [:math:`-`]

    # Returns
        float: Propulsion efficiency over maximum propulsion efficiency, [:math:`-`]
    """
    c_t_over_c_t_eta_b = c_t / c_t_eta_b

    sigma = np.where(mach_num < 0.4, 1.3 * (0.4 - mach_num), 0.0)

    eta_over_eta_b_low = (
        10.0 * (1.0 + 0.8 * (sigma - 0.43) - 0.6027 * sigma * 0.43) * c_t_over_c_t_eta_b
        + 33.3333
        * (-1.0 - 0.97 * (sigma - 0.43) + 0.8281 * sigma * 0.43)
        * (c_t_over_c_t_eta_b**2)
        + 37.037
        * (1.0 + (sigma - 0.43) - 0.9163 * sigma * 0.43)
        * (c_t_over_c_t_eta_b**3)
    )
    eta_over_eta_b_hi = (
        (1.0 + (sigma - 0.43) - sigma * 0.43)
        + (4.0 * sigma * 0.43 - 2.0 * (sigma - 0.43)) * c_t_over_c_t_eta_b
        + ((sigma - 0.43) - 6 * sigma * 0.43) * (c_t_over_c_t_eta_b**2)
        + 4.0 * sigma * 0.43 * (c_t_over_c_t_eta_b**3)
        - sigma * 0.43 * (c_t_over_c_t_eta_b**4)
    )
    return np.where(c_t_over_c_t_eta_b < 0.3, eta_over_eta_b_low, eta_over_eta_b_hi)


def thrust_coefficient_at_max_efficiency(
    mach_num: float, m_des: float, c_t_des: float
) -> float:
    """Calculate thrust coefficient at maximum overall propulsion efficiency for a given Mach Number.

    # Parameters
        mach_num (float):
            Mach number, [:math:`-`]
        m_des (float):
            Design Mach number, [:math:`-`]
        c_t_des (float):
            Thrust coefficient at design Mach number, [:math:`-`]

    # Returns
        float: Thrust coefficient at maximum overall propulsion efficiency, [:math:`-`]
    """
    m_over_m_des = mach_num / m_des
    h_2 = ((1.0 + 0.55 * mach_num) / (1.0 + 0.55 * m_des)) / (m_over_m_des**2)

    return h_2 * c_t_des


def max_overall_propulsion_efficiency(
    mach_num: float, eta_1: float, eta_2: float
) -> float:
    """Calculate maximum overall propulsion efficiency.

    # Parameters
        mach_num (float):
            Mach number, [:math:`-`]
        eta_1 (float):
            Efficiency parameter 1, [:math:`-`]
        eta_2 (float):
            Efficiency parameter 2, [:math:`-`]

    # Returns
        float: Maximum overall propulsion efficiency, [:math:`-`]
    """
    return eta_1 * mach_num**eta_2


# -------------------
# Full comsumption
# -------------------


def fuel_mass_flow_rate(
    air_pressure: float,
    air_temperature: float,
    mach_num: float,
    c_t: float,
    eta: float,
    wing_surface_area: float,
    q_fuel: float,
) -> float:
    """Calculate fuel mass flow rate.

    # Parameters
        air_pressure (float):
            Air pressure, [:math:`Pa`]
        air_temperature (float):
            Air temperature, [:math:`K`]
        mach_num (float):
            Mach number, [:math:`-`]
        c_t (float):
            Thrust coefficient, [:math:`-`]
        eta (float):
            Engine efficiency, [:math:`-`]
        wing_surface_area (float):
            Wing surface area, [:math:`m^2`]
        q_fuel (float):
            Fuel heating value, [:math:`J/kg`]

    # Returns
        float: Fuel mass flow rate, [:math:`kg/s`]
    """
    return (
        (constants.gamma / 2)
        * (c_t * mach_num**3 / eta)
        * (constants.gamma * constants.R * air_temperature * air_pressure) ** 0.5
        * air_pressure
        * wing_surface_area
        / q_fuel
    )


def fuel_flow_correction(
    fuel_flow: float,
    altitude_ft: float,
    air_temperature: float,
    air_pressure: float,
    mach_num: float,
    fuel_flow_idle_sls: float,
    fuel_flow_max_sls: float,
    flight_phase: str,
) -> float:
    """Correct fuel flow.

    # Parameters
        fuel_flow (float):
            Fuel flow, [:math:`kg/s`]
        altitude_ft (float):
            Altitude, [:math:`ft`]
        air_temperature (float):
            Air temperature, [:math:`K`]
        air_pressure (float):
            Air pressure, [:math:`Pa`]
        mach_num (float):
            Mach number, [:math:`-`]
        fuel_flow_idle_sls (float):
            Fuel flow at idle at sea level, [:math:`kg/s`]
        fuel_flow_max_sls (float):
            Maximum fuel flow at sea level, [:math:`kg/s`]
        flight_phase (str):
            Flight phase, [:math:`-`]

    # Returns
        float: Corrected fuel flow, [:math:`kg/s`]
    """

    ff_min = op_lim.fuel_flow_idle(fuel_flow_idle_sls, altitude_ft)
    ff_max = jet.equivalent_fuel_flow_rate_at_cruise(
        fuel_flow_max_sls,
        (air_temperature / constants.T_msl),
        (air_pressure / constants.p_surface),
        mach_num,
    )

    if flight_phase == "descent":
        ff_max = 0.3 * fuel_flow_max_sls

    return np.clip(fuel_flow, ff_min, ff_max)
