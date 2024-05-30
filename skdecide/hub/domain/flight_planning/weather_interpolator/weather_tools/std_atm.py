#!/usr/bin/python
# -*- coding: utf-8 -*-

# #############################################################################
# Copyright (c) 2008, Kevin Horton
# All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# *
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The name of Kevin Horton may not be used to endorse or promote products
#       derived from this software without specific prior written permission.
# *
# THIS SOFTWARE IS PROVIDED BY KEVIN HORTON ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
# EVENT SHALL KEVIN HORTON BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# #############################################################################
#
# version 0.16, 06 May 2007
#
# Version History:
# vers     date     Notes
#  0.1   14 May 06  First release.
#
# 0.11   17 May 06  Cleaned up documentation.
#
# 0.12   24 May 06  Added temp2speed_of_sound function.
#
# 0.13   02 Jun 06  Added temp2isa and isa2temp functions.
#
# 0.14   22 Apr 07  Added density altitude vs temperature functions
#
# 0.15   29 Apr 07  Broke out sat_press and dry_press as public functions.
#
# 0.16   05 May 07  Reworked to use default units from default_units module.
# #############################################################################
#
# To Do: 1. Done.
#
#        2. Done.
#
#        3. Done.
#
#        4. Won't do.
#
#        5. Add temp2temp_ratio, press2press_ratio and density2density_ratio.
#
#        6. Done.
#
#        7. Add tests for all functions to test/test_std_atm.py
#           Tests to add:
#           dry_press
#           density_alt_table ? (probably won't make a test for this)
#
#        8. Review code to catch duplicated code blocks in different functions.
#           Move these blocks to internal functions.
#
#        9. Review API for all public functions for consistency of units, etc.
#
# Done:  1. consider replacing calculations by constants where possible.  See
#           http://jcoppens.com/globo/teoria/atm_est.en.php
#
#           Tested replacing calculations by constants in press2alt.  The
#           perf improvement was only about 15%.  Probably not worth the trouble.
#           Better to keep the pedigree of the equations visible.
#
#        2. Validate against published data.  Created unittests using data:
#           http://www.sworld.com.au/steven/space/atmosphere/
#
#        3. Added relative humidity to density altitude calc.  See:
#           http://wahiduddin.net/calc/density_altitude.htm
#
#        4. Change formulae to use pressure in pa, not in HG.  Won't do.
#           Instead, changed to use default units specified in default_units.py
#
#        6. Added functions:
#           isa2temp
#           temp2isa
#
#        7. Added tests for functions:
#           pressure_alt
#           sat_press
#           density_alt2temp
# #############################################################################

"""Calculate standard atmosphere parametres.

Calculates standard atmosphere parametres, using the 1976 International
Standard Atmosphere.  The default units for the input and output are defined
in default_units.py

All altitudes are geopotential altitudes (i.e. it is assumed that there is
no variation with altitude of the acceleration due to gravity).

Works up to 84.852 km (278,386 ft) altitude.

"""

import locale as L
import math as M

import skdecide.hub.domain.flight_planning.weather_interpolator.weather_tools.unit_conversion as U

try:
    from default_units import *
except ImportError:
    default_area_units = "ft**2"
    default_power_units = "hp"
    default_speed_units = "kt"
    default_temp_units = "C"
    default_weight_units = "lb"
    default_press_units = "in HG"
    default_density_units = "lb/ft**3"
    default_length_units = "ft"
    default_alt_units = default_length_units
    default_avgas_units = "lb"

try:
    L.setlocale(L.LC_ALL, "en_US")
except:
    pass

g = 9.80665  # Acceleration of gravity at 45.542 deg latitude, m/s**s
Rd = 287.05307  # Gas constant for dry air, J/kg K

# conditions starting at sea level, in a region with temperature gradient

T0 = 288.15  # Temperature at sea level, degrees K
L0 = -6.5  # Temperature lapse rate, at sea level deg K/km
P0 = 29.9213  # Pressure at sea level, in HG
Rho0 = 1.2250  # Density at sea level, kg/m**3

# conditions starting at 11 km, in an isothermal region

T11 = T0 + 11 * L0  # Temperature at 11,000 m, degrees K
PR11 = (T11 / T0) ** ((-1000 * g) / (Rd * L0))  # pressure ratio at 11,000 m
P11 = PR11 * P0
Rho11 = (Rho0 * PR11) * (T0 / T11)

# conditions starting at 20 km, in a region with temperature gradient

T20 = T11
PR20 = PR11 * M.exp(((-1000 * g) * (20 - 11)) / (Rd * T11))
L20 = 1  # temperature lapse rate, starting at 20,000 m, deg K/km
P20 = PR20 * P0
Rho20 = (Rho0 * PR20) * (T0 / T20)

# conditions starting at 32 km, in a region with temperature gradient

T32 = 228.65  # Temperature at 32 km, degrees K
PR32 = PR20 * (T32 / T20) ** ((-1000 * g) / (Rd * L20))

# PR32 = PR20 * M.exp((-1000 * g) * (32 - 20)/(R * T20))

L32 = 2.8  # temperature lapse rate, starting at 32,000 m, deg K/km
P32 = PR32 * P0
Rho32 = (Rho0 * PR32) * (T0 / T32)

# conditions starting at 47 km, in an isothermal region

T47 = 270.65
PR47 = PR32 * (T47 / T32) ** ((-1000 * g) / (Rd * L32))
P47 = PR47 * P0
Rho47 = (Rho0 * PR47) * (T0 / T47)

# conditions starting at 51 km, in a region with temperature gradient

T51 = 270.65  # Temperature at 51 km, degrees K
PR51 = PR47 * M.exp(((-1000 * g) * (51 - 47)) / (Rd * T47))
L51 = -2.8  # temperature lapse rate, starting at 51,000 m, deg K/km
P51 = PR51 * P0
Rho51 = (Rho0 * PR51) * (T0 / T51)

# conditions starting at 71 km, in a region with temperature gradient

T71 = 214.65  # Temperature at 71 km, degrees K
PR71 = PR51 * (T71 / T51) ** ((-1000 * g) / (Rd * L51))
L71 = -2.0  # temperature lapse rate, starting at 71,000 m, deg K/km
P71 = PR71 * P0
Rho71 = (Rho0 * PR71) * (T0 / T71)

# temp_units_list = ['C', 'F', 'K', 'R']

# #############################################################################
#
# Altitude to temperature
#
# #############################################################################


def alt2temp(H, alt_units=default_alt_units, temp_units=default_temp_units):
    """Return the standard temperature for the specified altitude.  Altitude
    units may be feet ('ft'), metres ('m'), statute miles, ('sm') or
    nautical miles ('nm').  Temperature units may be degrees C, F, K or R
    ('C', 'F', 'K' or 'R')

    If the units are not specified, the units in default_units.py are used.

    Examples:

    Calculate the standard temperature (in default temperature units) at
    5,000 (default altitude units):
    >>> alt2temp(5000)
    5.0939999999999941

    Calculate the standard temperature in deg F at sea level:
    >>> alt2temp(0, temp_units = 'F')
    59.0

    Calculate the standard temperature in deg K at 11,000 m:
    >>> alt2temp(11000, alt_units = 'm', temp_units = 'K')
    216.64999999999998

    Calculate the standard temperature at 11 statute miles in deg R:
    >>> alt2temp(11, alt_units = 'sm', temp_units = 'R')
    389.96999999999997

    The input value may be an expression:
    >>> alt2temp(11 * 5280, temp_units = 'R')
    389.96999999999997

    """

    # Validated to 84000 m
    # uses meters and degrees K for the internal calculations

    # function tested in tests/test_std_atm.py

    H = U.len_conv(H, from_units=alt_units, to_units="km")

    if H <= 11:
        temp = T0 + H * L0
    elif H <= 20:
        temp = T11
    elif H <= 32:
        temp = T20 + (H - 20) * L20
    elif H <= 47:
        temp = T32 + (H - 32) * L32
    elif H <= 51:
        temp = T47
    elif H <= 71:
        temp = T51 + (H - 51) * L51
    elif H <= 84.852:
        temp = T71 + (H - 71) * L71
    else:
        raise ValueError(
            "This function is only implemented for altitudes of 84.852 km and below."
        )

    return U.temp_conv(temp, to_units=temp_units, from_units="K")


def alt2temp_ratio(H, alt_units=default_alt_units):
    """
    Return the temperature ratio (temperature / standard temperature for
    sea level).  The altitude is specified in feet ('ft'), metres ('m'),
    statute miles, ('sm') or nautical miles ('nm').

    If the units are not specified, the units in default_units.py are used.

    Examples:

    Calculate the temperature ratio at 8,000 (default altitude units)
    >>> alt2temp_ratio(8000)
    0.94499531494013533

    Calculate the temperature ratio at 8,000 m.
    >>> alt2temp_ratio(8000, alt_units = 'm')
    0.81953843484296374
    """

    # function tested in tests/test_std_atm.py

    return alt2temp(H, alt_units, temp_units="K") / T0


# #############################################################################
#
# ISA deviation to temperature
#
# #############################################################################


def isa2temp(
    ISA_dev,
    altitude,
    temp_units=default_temp_units,
    alt_units=default_alt_units,
):
    """
    Return the temperature that is a specified amount warmer or cooler
    than the standard temperature for the altitude.

    The temperature may be in deg C, F, K or R.

    The altitude may be in feet ('ft'), metres ('m'), kilometres ('km'),
    statute miles, ('sm') or nautical miles ('nm').

    If the units are not specified, the units in default_units.py are used.

    Examples:

    Determine the temperature that is 10 deg (default temperature units) warmer
    than the standard temperature at 8,000 (default altitude units):
    >>> isa2temp(10, 8000)
    9.1503999999999905

    Determine the temperature that is 25 degrees K cooler than the standard
    temperature at 2000 m.
    >>> isa2temp(-25, 2000, temp_units = 'K', alt_units = 'm')
    250.14999999999998
    """

    # function tested in tests/test_std_atm.py

    temp = ISA_dev + alt2temp(altitude, alt_units, temp_units)

    return temp


# #############################################################################
#
# temperature to ISA deviation
#
# #############################################################################


def temp2isa(
    temp,
    altitude,
    temp_units=default_temp_units,
    alt_units=default_alt_units,
):
    """
    Return the amount that the specified temperature is warmer or cooler
    than the standard temperature for the altitude.

    The temperature may be in deg C, F, K or R.

    The altitude may be in feet ('ft'), metres ('m'), kilometres ('km'),
    statute miles, ('sm') or nautical miles ('nm').

    If the units are not specified, the units in default_units.py are used.

    Examples:

    Determine the ISA deviation for a temperature of 30 deg (default
    temperature units) at an altitude of 2000 (default altitude units):
    >>> temp2isa(30, 2000)
    18.962400000000002

    Determine the ISA deviation in degrees F for a temperature of 45 deg F
    at an altitude of 1000 m:
    >>> temp2isa(45, 1000, temp_units = 'F', alt_units = 'm')
    -2.2999999999999972
    """

    # function tested in tests/test_std_atm.py

    std_temp = alt2temp(altitude, alt_units, temp_units)
    ISA_dev = temp - std_temp

    return ISA_dev


# #############################################################################
#
# Altitude to pressure and pressure ratio
#
# #############################################################################


def _alt2press_ratio_gradient(
    H,
    Hb,
    Pb,
    Tb,
    L,
):

    # eqn from USAF TPS PEC binder, page PS1-31

    return (Pb / P0) * (1 + (L / Tb) * (H - Hb)) ** ((-1000 * g) / (Rd * L))


def _alt2press_ratio_isothermal(
    H,
    Hb,
    Pb,
    Tb,
):

    # eqn from USAF TPS PEC binder, page PS1-26

    return (Pb / P0) * M.exp((-1 * (H - Hb)) * ((1000 * g) / (Rd * Tb)))


def alt2press_ratio(H, alt_units=default_alt_units):
    """
    Return the pressure ratio (atmospheric pressure / standard pressure
    for sea level).  The altitude is specified in feet ('ft'), metres ('m'),
    statute miles, ('sm') or nautical miles ('nm').

    If the units are not specified, the units in default_units.py are used.

    Examples:

    Calculate the pressure ratio at 5000 (default altitude units):
    >>> alt2press_ratio(5000)
    0.8320481158727735

    Calculate the pressure ratio at 1000 m:
    >>> alt2press_ratio(1000, alt_units = 'm')
    0.88699304638887044

    The functions are only implemented at altitudes of 84.852 km and lower.
    >>> alt2press_ratio(90, alt_units = 'km')
    Traceback (most recent call last):
      File '<stdin>', line 1, in ?
      File './std_atm.py', line 189, in alt2press_ratio
    if H <= 20:
    ValueError: This function is only implemented for altitudes of 84.852 km and below.
    """

    # uses meters and degrees K for the internal calculations

    # function tested in tests/test_std_atm.py

    H = U.len_conv(H, from_units=alt_units, to_units="km")

    if H <= 11:
        return _alt2press_ratio_gradient(H, 0, P0, T0, L0)
    if H <= 20:
        return _alt2press_ratio_isothermal(H, 11, P11, T11)
    if H <= 32:
        return _alt2press_ratio_gradient(H, 20, P20, T20, L20)
    if H <= 47:
        return _alt2press_ratio_gradient(H, 32, P32, T32, L32)
    if H <= 51:
        return _alt2press_ratio_isothermal(H, 47, P47, T47)
    if H <= 71:
        return _alt2press_ratio_gradient(H, 51, P51, T51, L51)
    if H <= 84.852:
        return _alt2press_ratio_gradient(H, 71, P71, T71, L71)
    else:
        raise ValueError(
            "This function is only implemented for altitudes of 84.852 km and below."
        )


def alt2press(H, alt_units=default_alt_units, press_units=default_press_units):
    """
    Return the atmospheric pressure for a given altitude, with the
    altitude in feet ('ft'), metres ('m'), statute miles, ('sm') or nautical
    miles ('nm'), and the pressure in inches of HG ('in HG'), mm of HG
    ('mm HG'), psi, lb per sq. ft ('psf'), pa, hpa or mb.

    If the units are not specified, the units in default_units.py are used.

    Examples:

    Calculate the pressure in inches of mercury at 5,000 (default altitude
    units):
    >>> alt2press(5000)
    24.895961289464015

    Calculate the pressure in pounds per square foot at 10,000 (default
    altitude units):
    >>> alt2press(10000, press_units = 'psf')
    1455.3301392981359

    Calculate the pressure in pascal at 20 km:
    >>> alt2press(20, press_units = 'pa', alt_units = 'km')
    5474.8827144576408
    """

    # uses meters, inches of HG and degrees K for the internal calculations

    # function tested in tests/test_std_atm.py

    H = U.len_conv(H, from_units=alt_units, to_units="m")

    press = P0 * alt2press_ratio(H, alt_units="m")
    press = U.press_conv(press, from_units="in HG", to_units=press_units)

    return press


# #############################################################################
#
# Pressure altitude from barometric altitude and altimeter setting
#
# #############################################################################


def pressure_alt(H, alt_setting, alt_units=default_alt_units):
    """
    Return the pressure altitude, given the barometric altitude and the
    altimeter setting.

    Altimeter setting may have units of inches of HG, or hpa or mb.  If the
    altimeter setting value is less than 35, the units are assumed to be
    in HG, otherwise they are assumed to be hpa.  The altimeter setting must
    be in the range of 25 to 35 inches of mercury.

    The altitude may have units of feet ('ft'), metres ('m'), statute miles,
    ('sm') or nautical miles ('nm').

    If the units are not specified, the units in default_units.py are used.

    Examples:

    Calculate the pressure altitude for 1,000 (default altitude units)
    barometric altitude with altimeter setting of 30.92 in HG:
    >>> pressure_alt(1000, 30.92)
    88.612734282205338

    Calculate the pressure altitude for 1,000 (default altitude units)
    barometric altitude with altimeter setting of 1008 mb:
    >>> pressure_alt(1000, 1008)
    1143.6503495627171

    Calculate the pressure altitude in metres for 304.8 m barometric
    altitude with altimeter setting of 1008 mb:
    >>> pressure_alt(304.8, 1008, alt_units = 'm')
    348.58462654671621
    """

    H = U.len_conv(H, from_units=alt_units, to_units="ft")
    if alt_setting > 35:
        alt_setting = U.press_conv(alt_setting, from_units="hpa", to_units="in HG")
    if alt_setting < 25 or alt_setting > 35:
        raise ValueError("Altimeter setting out of range.")
    HP = H + 145442.2 * (1 - (alt_setting / P0) ** 0.190261)
    HP = U.len_conv(HP, from_units="ft", to_units=alt_units)
    return HP


def QNH(
    HP,
    H,
    alt_units=default_alt_units,
    alt_setting_units="in HG",
):
    """
    Return the altimeter setting, given the pressure altitude (HP) and the
    barometric altitude (H).
    """

    HP = U.len_conv(HP, from_units=alt_units, to_units="ft")
    H = U.len_conv(H, from_units=alt_units, to_units="ft")
    QNH = P0 * (1 - (HP - H) / 145442.2) ** 5.255594
    QNH = U.press_conv(QNH, from_units="in HG", to_units=alt_setting_units)

    return QNH


# #############################################################################
#
# Altitude to density and density ratio
#
# #############################################################################


def alt2density_ratio(H, alt_units=default_alt_units):
    """
    Return the density ratio (atmospheric density / standard density
    for sea level).  The altitude is specified in feet ('ft'), metres ('m'),
    statute miles, ('sm') or nautical miles ('nm').

    If the units are not specified, the units in default_units.py are used.

    Examples:

    Calculate the density ratio at 7,500 (default altitude units):
    >>> alt2density_ratio(7500)
    0.79825819881753035

    Calculate the density ratio at 2 km:
    >>> alt2density_ratio(2, alt_units = 'km')
    0.8216246960994622
    """

    # function tested in tests/test_std_atm.py

    return alt2press_ratio(H, alt_units) / alt2temp_ratio(H, alt_units)


def alt2density(H, alt_units=default_alt_units, density_units=default_density_units):
    """
    Return the density given the pressure altitude.  The altitude is
    specified in feet ('ft'), metres ('m'), statute miles, ('sm') or
    nautical miles ('nm').

    The desired density units are specified as 'lb/ft**3', 'slug/ft**3' or
    'kg/m**3'.

    If the units are not specified, the units in default_units.py are used.

    Examples:

    Calculate the density in lb / ft cubed at 7,500 (default altitude units):
    >>> alt2density(7500)
    0.061046199847730374

    Calculate the density in slugs / ft cubed at 5,000 (default altitude units):
    >>> alt2density(5000, density_units = 'slug/ft**3')
    0.0020480982157718704

    Calculate the density in kg / m cubed at 0 (default altitude units:
    >>> alt2density(0, density_units = 'kg/m**3')
    1.2250000000000001

    Calculate the density in kg / m cubed at 81,000 m:
    >>> alt2density(81000, density_units = 'kg/m**3', alt_units = 'm')
    1.3320480184052337e-05
    """

    # function tested in tests/test_std_atm.py

    # get density in kg/m**3

    density = Rho0 * alt2density_ratio(H, alt_units)
    return U.density_conv(density, from_units="kg/m**3", to_units=density_units)


# #############################################################################
#
# Density to altitude and density ratio to altitude
#
# #############################################################################


def _density2alt_gradient(
    Rho,
    Rhob,
    Hb,
    Tb,
    L,
):

    return Hb + (Tb / L) * ((Rho / Rhob) ** (-1 / ((1000 * g) / (Rd * L) + 1)) - 1)


def _density2alt_isothermal(
    Rho,
    Rhob,
    Hb,
    Tb,
):

    return Hb - ((Rd * Tb) * M.log(Rho / Rhob)) / (1000 * g)


def density2alt(Rho, density_units=default_density_units, alt_units=default_alt_units):
    """
    Return the altitude corresponding to the specified density, with
    density in 'lb/ft**3', 'slug/ft**3' or 'kg/m**3'.

    The altitude is specified in feet ('ft'), metres ('m'), statute miles,
    ('sm') or nautical miles ('nm').

    If the units are not specified, the units in default_units.py are used.

    Examples:

    Calculate the altitude in default altitude units where the density is
    0.056475 in default density units:
    >>> density2alt(.056475)
    9999.8040934937271

    Calculate the altitude in metres where the density is 0.018012 kg / m
    cubed:
    >>> density2alt(.018012, alt_units = 'm', density_units = 'kg/m**3')
    29999.978688508152
    """

    # function tested in tests/test_std_atm.py

    Rho = U.density_conv(Rho, from_units=density_units, to_units="kg/m**3")

    if Rho > Rho11:
        H = _density2alt_gradient(Rho, Rho0, 0, T0, L0)
    elif Rho > Rho20:
        H = _density2alt_isothermal(Rho, Rho11, 11, T11)
    elif Rho > Rho32:
        H = _density2alt_gradient(Rho, Rho20, 20, T20, L20)
    elif Rho > Rho47:
        H = _density2alt_gradient(Rho, Rho32, 32, T32, L32)
    elif Rho > Rho51:
        H = _density2alt_isothermal(Rho, Rho47, 47, T47)
    elif Rho > Rho71:
        H = _density2alt_gradient(Rho, Rho51, 51, T51, L51)
    else:
        H = _density2alt_gradient(Rho, Rho71, 71, T71, L71)

    if H > 84.852:
        raise ValueError(
            "This function is only implemented for altitudes of 84.852 km and below."
        )

    return U.len_conv(H, from_units="km", to_units=alt_units)


def density_ratio2alt(DR, alt_units=default_alt_units):
    """
    Return the altitude for the specified density ratio. The altitude is in
    feet ('ft'), metres ('m'), statute miles, ('sm') or nautical miles
    ('nm').

    If the units are not specified, the units in default_units.py are used.

    Examples:

    Calculate the altitude in default altitude units where the density ratio is
    1:
    >>> density_ratio2alt(1)
    0.0

    Calculate the altitude in feet where the density ratio is 0.5:
    >>> density_ratio2alt(.5)
    21859.50324995652

    Calculate the altitude in km where the density ratio is 0.1
    >>> density_ratio2alt(.1, alt_units = 'km')
    17.9048674520646
    """

    # function tested in tests/test_std_atm.py

    D = DR * Rho0
    return density2alt(D, alt_units=alt_units, density_units="kg/m**3")


# #############################################################################
#
# Density Altitude
#
# #############################################################################


def density_alt(
    H,
    T,
    alt_setting=P0,
    DP="FALSE",
    RH=0.0,
    alt_units=default_alt_units,
    temp_units=default_temp_units,
):
    """
    Return density altitude, given the pressure altitude and the
    temperature with altitudes in units of feet ('ft'), metres ('m'),
    statute miles, ('sm') or nautical miles ('nm'), and temperature in units
    of deg C, F, K or R ('C', 'F', 'K' or 'R').

    Mandatory parametres:
    H = altitude
    T = temperature

    Optional parametres:
    alt_setting = altimeter setting (defaults to 29.9213 if not provided
    DP = dew point
    RH = relative humidity
    alt_units = units for the altitude.  'ft', 'm', or 'km'.
    temp_units = units for the temperature and dew point.  'C', 'F', 'K'
                 or 'R'.

    The altimeter setting units are assumed to be inches of HG, unless the
    value is greater than 35.  In this case the units are assumed to be mb.

    If the dew point or relative humidity are not specified, the air is
    assumed to be completely dry.  If both the dew point and relative humidity
    are specified, the relative humidity value is ignored.

    If the units are not specified, the units in default_units.py are used.

    The method is from: http://wahiduddin.net/calc/density_altitude.htm

    Examples:

    Calculate the density altitude in default altitude units for a pressure
    altitude of 7000 default altitude units and a temperature of 15 deg
    (default temperature units).  The altimeter setting is not specified, so it
    defaults to standard pressure of 29.9213 in HG or 1013.25 mb:
    >>> density_alt(7000, 15)
    8595.3465863232504

    Calculate the density altitude in default altitude units for a pressure
    altitude of 7000 default altitude units and a temperature of 85 deg F.
    The altimeter setting is not specified, so it defaults to standard pressure
    of 29.9213 in HG or 1013.25 mb.  The dew point and relative humidity are
    not specified, so the air is assumed to be dry:
    >>> density_alt(7000, 85, temp_units = 'F')
    10159.10696106757

    Calculate the density altitude in default altitude units for a pressure
    altitude of 7000 default altitude units, an altimeter setting of 29.80 and
    a temperature of 85 deg F and a dew point of 55 deg F:
    >>> density_alt(7000, 85, 29.80, 55, temp_units = 'F')
    10522.776013011618

    Calculate the density altitude in metres for a pressure altitude of
    2000 m, an altimeter setting of 1010 mb,  a temperature of 15 deg (default
    temperature units) and a relative humidity of 50%:
    >>> density_alt(2000, 15, 1010, alt_units = 'm', RH = 0.5)
    2529.8230634449737

    The dew point may be specified in one of two ways: as the fourth
    argument on the command line, or via the keyword argument DP.
    >>> density_alt(2000, 15, 1010, alt_units = 'm', DP = 5)
    2530.7528237990618

    The relative humidity must be in the range of 0 to 1:
    >>> density_alt(2000, 15, 1010, alt_units = 'm', RH = 1.1)
    Traceback (most recent call last):
      File '<stdin>', line 1, in ?
      File 'std_atm.py', line 533, in density_alt
    raise ValueError, 'The relative humidity must be in the range of 0 to 1.'
    ValueError: The relative humidity must be in the range of 0 to 1.
    """

    Rv = 461.495  # gas constant for water vapour

    # saturated vapour pressure

    if DP == "FALSE" and RH == 0:
        Pv = 0
    else:
        Pv = sat_press(T, DP, RH, temp_units, press_units="pa")

    # dry air pressure

    Pd = dry_press(
        H, Pv, alt_setting=alt_setting, alt_units=alt_units, press_units="pa"
    )

    T = U.temp_conv(T, from_units=temp_units, to_units="K")
    D = Pd / (Rd * T) + Pv / (Rv * T)

    DR = D / Rho0

    return density_ratio2alt(DR, alt_units)


def _sat_press(T):
    """
    Return the saturation pressure in mb of the water vapour, given
    temperature in deg C.  Equation from:
    http://wahiduddin.net/calc/density_altitude.htm
    """

    eso = 6.1078
    c0 = 0.99999683
    c1 = -0.90826951e-2
    c2 = 0.78736169e-4
    c3 = -0.61117958e-6
    c4 = 0.43884187e-8
    c5 = -0.29883885e-10
    c6 = 0.21874425e-12
    c7 = -0.17892321e-14
    c8 = 0.11112018e-16
    c9 = -0.30994571e-19

    p = c0 + T * (
        c1
        + T
        * (
            c2
            + T * (c3 + T * (c4 + T * (c5 + T * (c6 + T * (c7 + T * (c8 + T * c9))))))
        )
    )
    sat_press = eso / p**8
    return sat_press


def sat_press(
    T="FALSE",
    DP="FALSE",
    RH=0.0,
    temp_units=default_temp_units,
    press_units=default_press_units,
):
    """
    Return the saturated vapour pressure of water.  Either the dew point, or
    the temperature and the relative humidity must be specified.  If both the
    dew point and relative humidity are specified, the relative humidity value
    is ignored.

    If the temperature and dew point are both specified, the dew point cannot
    be greater than the temperature:

    If the units are not specified, the units in default_units.py are used.

    >>> sat_press(T=10, DP=11)
    Traceback (most recent call last):
      File '<stdin>', line 1, in <module>
      File 'std_atm.py', line 795, in sat_press
        raise ValueError, 'The dew point cannot be greater than the temperature.'
    ValueError: The dew point cannot be greater than the temperature.

    Dew point is 11 deg (default temperature units).  Find the water vapour
    pressure in default pressure units:
    >>> sat_press(DP=11)
    0.38741015927568667

    Dew point is 65 deg F.  Find the water vapour pressure in default pressure units:
    >>> sat_press(DP=65, temp_units = 'F')
    0.62207710701956165

    Dew point is 212 deg F (the boiling point of water at sea level).
    Find the water vapour pressure in lb per sq. inch:
    >>> sat_press(DP=212, temp_units = 'F', press_units = 'psi')
    14.696764873564959

    Temperature is 30 deg C.  Find the water vapour pressure in default pressure units:
    for 50% relative humidity:
    >>> sat_press(T=30, RH = 0.5)
    0.62647666996057927
    """

    if DP != "FALSE":

        # use dew point method

        if T != "FALSE":
            if DP > T:
                raise ValueError(
                    "The dew point cannot be greater than the temperature."
                )

        DP = U.temp_conv(DP, from_units=temp_units, to_units="C")

        # calculate vapour pressure

        Pv = _sat_press(DP) * 100
    else:

        if RH == "FALSE":
            raise ValueError(
                "Either DP (dew point) or RH (relative humidity) must be specified."
            )

        # relative humidity is specified
        # confirm relative humidity is in range

        if RH < 0 or RH > 1:
            raise ValueError("The relative humidity must be in the range of 0 to 1.")

        if T == "FALSE":
            raise ValueError(
                "If the relative humidity is specified, the temperature must also be specified."
            )

        T = U.temp_conv(T, from_units=temp_units, to_units="C")

        Pv = _sat_press(T) * 100
        Pv *= RH

    Pv = U.press_conv(Pv, from_units="pa", to_units=press_units)

    return Pv


def dry_press(
    H,
    Pv,
    alt_setting=P0,
    alt_units=default_alt_units,
    press_units=default_press_units,
):
    """
    Returns dry air pressure, i.e. the total air pressure, less the water
    vapour pressure.
    """

    HP = pressure_alt(H, alt_setting, alt_units=alt_units)
    P = alt2press(HP, press_units=press_units, alt_units=alt_units)
    Pd = P - Pv

    return Pd


def density_alt2temp(
    density_alt_seek,
    press_alt,
    alt_units=default_alt_units,
    temp_units=default_temp_units,
):
    """
    Return temperature to achieve a desired density altitude.

    If the units are not specified, the units in default_units.py are used.
    """

    low = -100  # initial lower guess
    high = 100  # initial upper guess

    # confirm initial low and high are OK:

    da_low = density_alt(press_alt, low, alt_units=alt_units)
    if da_low > density_alt_seek:
        raise ValueError("Initial low guess too high.")

    da_high = density_alt(press_alt, high, alt_units=alt_units)
    if da_high < density_alt_seek:
        raise ValueError("Initial high guess too low.")

    guess = (low + high) / 2.0
    da_guess = density_alt(press_alt, guess, alt_units=alt_units)

    # keep iterating until da is within 1 ft of desired value

    while M.fabs(da_guess - density_alt_seek) > 1:
        if da_guess > density_alt_seek:
            high = guess
        else:
            low = guess

        guess = (low + high) / 2.0
        da_guess = density_alt(press_alt, guess, alt_units=alt_units)

    guess = U.temp_conv(guess, from_units="C", to_units=temp_units)

    return guess


def density_alt_table(
    density_alt_seek,
    alt_range=2000,
    alt_inc=100,
    alt_units=default_alt_units,
    temp_units=default_temp_units,
    multi_units=False,
    file="",
    format="text",
):
    """
    Return a text or html table of required temperature vs pressure altitude.

    If the units are not specified, the units in default_units.py are used.
    """

    line_buffer = []
    if format == "text":
        line_buffer.append("Pressure altitudes and temperatures for a density ")
        line_buffer.append("altitude of " + str(density_alt_seek) + " " + alt_units)
        line_buffer.append("(assuming dry air)\n")
        if multi_units:
            line_buffer.append(" Pressure    Temp      Temp")
            line_buffer.append(" Altitude")
            line_buffer.append("   (" + alt_units + ")     (deg C)   (deg F)")
        else:
            line_buffer.append(" Pressure    Temp")
            line_buffer.append(" Altitude")
            line_buffer.append("   (" + alt_units + ")     (deg " + temp_units + ")")
    elif format == "html":
        print("creating html")
    else:
        raise ValueError('Invalid format.  Must be either "text" or "html"')

    if multi_units:
        for alt in range(
            max(density_alt_seek - alt_range / 2.0, 0),
            density_alt_seek + alt_range / 2.0 + alt_inc,
            alt_inc,
        ):
            temp_c = density_alt2temp(density_alt_seek, alt, alt_units=alt_units)
            temp_f = U.temp_conv(temp_c, from_units="C", to_units="F")
            alt_str = L.format("%.*f", (0, alt), grouping=True)
            temp_c_str = "%.1f" % temp_c
            temp_f_str = "%.1f" % temp_f
            line_buffer.append(
                alt_str.rjust(6) + temp_c_str.rjust(11) + temp_f_str.rjust(10)
            )
    else:
        for alt in range(
            max(density_alt_seek - alt_range / 2.0, 0),
            density_alt_seek + alt_range / 2.0 + alt_inc,
            alt_inc,
        ):
            alt_str = L.format("%.*f", (0, alt), grouping=True)
            temp_str = "%.1f" % density_alt2temp(
                density_alt_seek, alt, temp_units=temp_units, alt_units=alt_units
            )
            line_buffer.append(alt_str.rjust(6) + temp_str.rjust(11))

    if file != "":
        OUT = open(file, "w")
        for line in line_buffer:
            OUT.write(line + "\n")

        print("file selected")
    else:
        return "\n".join(line_buffer)


# #############################################################################
#
# Pressure to altitude and pressure ratio to altitude
#
# #############################################################################


def _press2alt_gradient(
    P,
    Pb,
    Hb,
    Tb,
    L,
):

    return Hb + (Tb / L) * ((P / Pb) ** (((-1 * Rd) * L) / (1000 * g)) - 1)


def _press2alt_isothermal(
    P,
    Pb,
    Hb,
    Tb,
):

    return Hb - ((Rd * Tb) * M.log(P / Pb)) / (1000 * g)


def press2alt(P, press_units=default_press_units, alt_units=default_alt_units):
    """
    Return the altitude corresponding to the specified pressure, with
    pressure in inches of HG, mm of HG, psi, psf (lb per sq. ft), pa, hpa or
    mb.

    The altitude is in units of feet ('ft'), metres ('m'), statute miles,
    ('sm') or nautical miles ('nm')

    If the units are not specified, the units in default_units.py are used.

    Examples:

    Calculate the pressure altitude in feet for a pressure of 31.0185 inches
    of HG:
    >>> press2alt(31.0185)
    -999.98992888235091

    Calculate the pressure altitude in feet for a pressure of
    1455.33 lb sq. ft:
    >>> press2alt(1455.33, press_units = 'psf')
    10000.002466564831

    Calculate the pressure altitude in metres for a pressure of
    90.3415 mm HG:
    >>> press2alt(90.3415, press_units = 'mm HG', alt_units = 'm')
    15000.025465320754

    Calculate the pressure altitude in metres for a pressure of
    1171.86 pascal:
    >>> press2alt(1171.86, press_units = 'pa', alt_units = 'm')
    30000.029510365184
    """

    # function tested in tests/test_std_atm.py

    P = U.press_conv(P, from_units=press_units, to_units="in HG")

    if P > P11:
        H = _press2alt_gradient(P, P0, 0, T0, L0)
    elif P > P20:
        H = _press2alt_isothermal(P, P11, 11, T11)
    elif P > P32:
        H = _press2alt_gradient(P, P20, 20, T20, L20)
    elif P > P47:
        H = _press2alt_gradient(P, P32, 32, T32, L32)
    elif P > P51:
        H = _press2alt_isothermal(P, P47, 47, T47)
    elif P > P71:
        H = _press2alt_gradient(P, P51, 51, T51, L51)
    else:
        H = _press2alt_gradient(P, P71, 71, T71, L71)

    if H > 84.852:
        raise ValueError(
            "This function is only implemented for altitudes of 84.852 km and below."
        )

    return U.len_conv(H, from_units="km", to_units=alt_units)


def press_ratio2alt(PR, alt_units=default_alt_units):
    """
    Return the pressure ratio for the specified altitude.  The altitude is
    specified in feet ('ft'), metres ('m'), statute miles, ('sm') or
    nautical miles ('nm').

    If the units are not specified, the units in default_units.py are used.

    Examples:

    Calculate the altitude in feet where the pressure ratio is 0.5:
    >>> press_ratio2alt(.5)
    17969.990746028907

    Calculate the altitude in metres where the pressure ratio is 0.1:
    >>> press_ratio2alt(.1, alt_units = 'm')
    16096.249927559489
    """

    # function tested in tests/test_std_atm.py

    P = PR * P0
    return press2alt(P, alt_units=alt_units)


# #############################################################################
#
# Temperature to speed of sound
#
# #############################################################################


def temp2speed_of_sound(
    temp, temp_units=default_temp_units, speed_units=default_speed_units
):
    """
    Return the speed of sound, given the air temperature.

    The temperature units may be deg C, F, K or R ('C', 'F', 'K' or 'R').

    The speed units may be 'kt', 'mph', 'km/h', 'm/s' and 'ft/s'.

    If the units are not specified, the units in default_units.py are used.

    Examples:

    Determine speed of sound in knots at 15 deg (default temperature units):
    >>> temp2speed_of_sound(15)
    661.47882487301808

    Determine speed of sound in mph at 120 deg F:
    >>> temp2speed_of_sound(120, speed_units = 'mph', temp_units = 'F')
    804.73500154991291
    """

    # function tested in tests/test_std_atm.py

    temp = U.temp_conv(temp, from_units=temp_units, to_units="K")

    speed_of_sound = M.sqrt((1.4 * Rd) * temp)
    speed_of_sound = U.speed_conv(
        speed_of_sound, from_units="m/s", to_units=speed_units
    )

    return speed_of_sound
