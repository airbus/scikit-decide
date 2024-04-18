# data
import numpy as np

from skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.parameters import (
    constants,
)


def ft_to_pl(h: float) -> float:
    """
    Convert from altitude (ft) to pressure level (hPa).
    Assumes the ICAO standard atmosphere.
    """
    return m_to_pl(ft_to_m(h))


def ft_to_m(ft: float) -> float:
    """
    Convert length from feet to meter.
    """
    return ft * 0.3048


def m_to_pl(h):
    """
    Convert from altitude (m) to pressure level (hPa).
    """
    condlist = [h < constants.h_tropopause, h >= constants.h_tropopause]
    funclist = [
        _low_altitude_m_to_pl,
        _high_altitude_m_to_pl,
        np.nan,
    ]  # nan passed through
    return np.piecewise(h, condlist, funclist)


def _low_altitude_m_to_pl(h):
    T_isa: np.ndarray = m_to_T_isa(h)
    power_term = -constants.g / (constants.T_lapse_rate * constants.R)
    return (constants.p_surface * (T_isa / constants.T_msl) ** power_term) / 100.0


def _high_altitude_m_to_pl(h):
    T_tropopause_isa = m_to_T_isa(np.asarray(constants.h_tropopause))
    power_term = -constants.g / (constants.T_lapse_rate * constants.R)
    p_tropopause_isa = (
        constants.p_surface * (T_tropopause_isa / constants.T_msl) ** power_term
    )
    inside_exp = (-constants.g / (constants.R * T_tropopause_isa)) * (
        h - constants.h_tropopause
    )
    return p_tropopause_isa * np.exp(inside_exp) / 100.0


def m_to_T_isa(h):
    """
    Calculate the ambient temperature (K) for a given altitude (m).
    """

    h_min = np.minimum(h, constants.h_tropopause)

    return constants.T_msl + h_min * constants.T_lapse_rate


def tas_to_mach_number(true_airspeed, T):
    """
    Calculate Mach number from true airspeed at a specified ambient temperature.
    """

    return true_airspeed / np.sqrt((constants.gamma * constants.R) * (T + 1e-8))


def mach_number_to_tas(mach_number, T):
    """
    Calculate true airspeed from the Mach number at a specified ambient temperature.
    """
    return mach_number * np.sqrt((constants.gamma * constants.R) * T)
