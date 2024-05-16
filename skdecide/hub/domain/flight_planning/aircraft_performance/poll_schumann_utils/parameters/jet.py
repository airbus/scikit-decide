import numpy as np

from skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.parameters import (
    units,
)


def equivalent_fuel_flow_rate_at_cruise(
    fuel_flow_sls: float, theta_amb: float, delta_amb: float, mach_num: float
) -> float:
    """Convert fuel mass flow rate at sea level to equivalent fuel flow rate at cruise conditions.

    # Parameters
        fuel_flow_sls (float):
            Fuel mass flow rate at sea level, [:math:`kg/s`].
        theta_amb (float):
            Ambient temperature ratio, [:math:`K`].
        delta_amb (float):
            Ambient pressure ratio, [:math:`Pa`].
        mach_num (float):
            Mach number, [:math:`-`].

    # Returns
        float: Equivalent fuel mass flow rate at cruise conditions, [:math:`kg/s`].
    """

    denom = (theta_amb**3.8 / delta_amb) * np.exp(0.2 * mach_num**2)
    # denominator must be >= 1, otherwise corrected_fuel_flow  > fuel_flow_max_sls
    denom = np.clip(denom, 1.0, None)

    return fuel_flow_sls / denom


def clip_mach_number(
    true_airspeed: float,
    air_temperature: float,
    max_mach_number: float,
):
    """Compute the Mach number from the true airspeed and ambient temperature.

    # Parameters
        true_airspeed (float):
            True airspeed, [:math:`m/s`].
        air_temperature (float):
            Ambient temperature, [:math:`K`].
        max_mach_number (float):
            Maximum permitted operational Mach number for aircraft type.

    # Returns
        Tuple[np.ndarray, np.ndarray]: Adjusted true airspeed and Mach number.
    """

    mach_num = units.tas_to_mach_number(true_airspeed, air_temperature)

    is_unrealistic = mach_num > max_mach_number
    if not np.any(is_unrealistic):
        return true_airspeed, mach_num

    max_tas = units.mach_number_to_tas(max_mach_number, air_temperature)
    adjusted_mach_num = np.where(is_unrealistic, max_mach_number, mach_num)
    adjusted_true_airspeed = np.where(is_unrealistic, max_tas, true_airspeed)

    return adjusted_true_airspeed, adjusted_mach_num


def rate_of_climb_descent(
    altitude_current: float, altitude_next: float, delta_time: float
) -> float:
    """Compute the rate of climb or descent (ft/min) from the path angle and speed.

    # Parameters
        altitude_current (float):
            Current altitude, [:math:`ft`].
        altitude_next (float):
            Next altitude, [:math:`ft`].
        delta_time (float):
            Time step, [:math:`s`].

    # Returns
        float: Rate of climb or descent, [:math:`ft/min`].
    """

    delta_time_min = delta_time / 60.0

    return (altitude_next - altitude_current) / (delta_time_min + 1e-8)


# TODO: if speed not constant, change this function
def acceleration(speed_current, delta_time) -> float:
    """Calculate the acceleration/deceleration at each waypoint.

    # Parameters
        speed_current (float):
            Current speed, [:math:`kts`].
        delta_time (float):
            Time step, [:math:`s`].

    # Returns
        float: Acceleration, [:math:`kts/s`].
    """

    # here the speed is constant.

    return 0
