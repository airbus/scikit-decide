# math
import numpy as np

from skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.parameters import (
    constants,
)


def turbine_entry_temperature_at_max_take_off(first_flight: float) -> float:
    """Calculate turbine entry temperature at maximum take-off rating.

    # Parameters
        first_flight (float):
            Year of first flight of the aircraft.

    # Returns
        float: Turbine entry temperature at maximum take-off rating, [:math:`K`]
    """
    return 2000.0 * (1 - np.exp(62.8 - 0.0325 * first_flight))


def turbine_entry_temperature_at_max_continuous_climb(tet_mto: float) -> float:
    """Calculate turbine entry temperature at maximum continuous climb rating.

    # Parameters
        tet_mto (float):
            Turbine entry temperature at maximum take-off rating, [:math:`K`]

    # Returns
        float: Turbine entry temperature at maximum continuous climb rating, [:math:`K`]
    """
    return 0.92 * tet_mto


def impact_pressure_max_operating_limits(max_mach_num: float) -> float:
    """Calculate maximum permitted operational impact pressure.

    # Parameters
        max_mach_num (float):
            Maximum permitted operational Mach number for aircraft type.

    # Returns
        float: Maximum permitted operational impact pressure for aircraft type, ``p_i_max``, [:math:`Pa`]

    Notes:
        The impact pressure is the difference between the free stream total pressure ``p_0`` and the
        atmospheric static pressure ``p_inf``. By definition, the calibrated airspeed, ``v_cas``, is
        the speed at sea level in the ISA that has the same impact pressure.
    """
    v_cas_mo_over_c_msl = max_calibrated_airspeed_over_speed_of_sound(max_mach_num)
    return constants.p_surface * (
        (1.0 + 0.5 * (constants.gamma - 1.0) * v_cas_mo_over_c_msl**2)
        ** (constants.gamma / (constants.gamma - 1.0))
        - 1.0
    )


def max_calibrated_airspeed_over_speed_of_sound(max_mach_num: float) -> float:
    """Calculate max calibrated airspeed over the speed of sound at ISA mean sea level.

    # Parameters
        max_mach_num (float):
            Maximum permitted operational Mach number for aircraft type.

    # Returns
        float: Maximum calibrated airspeed over the speed of sound at ISA mean sea level, ``v_cas_mo_over_c_msl``
    """
    return 0.57 * (max_mach_num + 0.10)


def crossover_pressure_altitude(max_mach_num: float, p_i_max: float) -> float:
    """Calculate crossover pressure altitude.

    # Parameters
        max_mach_num (float):
            Maximum permitted operational Mach number for aircraft type.
        p_i_max (float):
            Maximum permitted operational impact pressure for aircraft type, [:math:`Pa`]

    # Returns
        float: Crossover pressure altitude, [:math:`ft`]
    """
    return p_i_max / (
        0.5
        * constants.gamma
        * max_mach_num**2
        * (1.0 + (max_mach_num**2 / 4.0) + (max_mach_num**4 / 40.0))
    )
