# data
# typing
from typing import Dict

import numpy as np

# utils
from skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.parameters import (
    constants,
)


def fuel_flow_idle(fuel_flow_idle_sls: float, altitude_ft: float) -> float:
    """
    Calculate minimum fuel mass flow rate at flight idle conditions.
    """

    x = altitude_ft / 10_000.0
    return fuel_flow_idle_sls * (1.0 - 0.178 * x + 0.0085 * x**2)


def max_mach_number_by_altitude(
    altitude_ft: float,
    air_pressure: float,
    max_mach_num: float,
    p_i_max: float,
    p_inf_co: float,
    *,
    atm_speed_limit: bool = True,
    buffer: float = 0.02,
) -> float:
    """
    Calculate maximum permitted Mach number at a given altitude.

    altitude_ft: current altitude (ft)
    air_pressure: current air pressure (Pa)
    max_mach_num: maximum Mach number
    p_i_max: maximum dynamic pressure (Pa)
    p_inf_co: critical pressure (Pa)
    atm_speed_limit: whether to apply atmospheric speed limit
    buffer: buffer for maximum Mach number
    """

    if atm_speed_limit:
        p_i_max = np.where(altitude_ft < 10000.0, 10510.0, p_i_max)  # type: ignore[assignment]

    return (
        np.where(  # type: ignore[return-value]
            air_pressure > p_inf_co,
            2.0**0.5
            * ((1.0 + (2.0 / constants.gamma) * (p_i_max / air_pressure)) ** 0.5 - 1.0)
            ** 0.5,
            max_mach_num,
        )
        + buffer
    )


def max_available_thrust_coefficient(
    air_temperature: float,
    mach_number: float,
    c_t_eta_b: float,
    aircraft_parameters: Dict[str, float],
    *,
    buffer: float = 0.05,
) -> float:
    """
    Calculate maximum available thrust coefficient.

    Parameters
    ----------
    air_temperature : ArrayOrFloat
        Ambient temperature at each waypoint, [:math:`K`]
    mach_number : ArrayOrFloat
        Mach number at each waypoint.
    c_t_eta_b : ArrayOrFloat
        Thrust coefficient at maximum overall propulsion efficiency for a given Mach Number.
    aircraft_parameters : PSAircraftEngineParams
        Extracted aircraft and engine parameters.
    buffer : float, optional
        Additional buffer for maximum available thrust coefficient. The default value is 0.05,
        which increases the maximum available thrust coefficient by 5%.

    Returns
    -------
    ArrayOrFloat
        Maximum available thrust coefficient that can be supplied by the engines.
    """
    tr_max = _normalised_max_throttle_parameter(
        air_temperature,
        mach_number,
        aircraft_parameters["tet_mcc"],
        aircraft_parameters["tr_ec"],
        aircraft_parameters["m_ec"],
    )
    c_t_max_over_c_t_eta_b = 1.0 + 2.5 * (tr_max - 1.0)
    return c_t_max_over_c_t_eta_b * c_t_eta_b * (1.0 + buffer)


def _normalised_max_throttle_parameter(
    air_temperature: float,
    mach_number: float,
    tet_mcc: float,
    tr_ec: float,
    m_ec: float,
) -> float:
    """
    Calculate normalised maximum throttle parameter.
    """

    return (tet_mcc / air_temperature) / (
        tr_ec
        * (1.0 - 0.53 * (mach_number - m_ec) ** 2)
        * (1.0 + 0.2 * mach_number**2)
    )
