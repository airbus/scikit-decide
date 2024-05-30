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
# version 0.22, 25 Apr 2008
#
# Version History:
# vers     date     Notes
#  0.1   14 May 06  First release. Only has the unit conversions needed
#                   for the std_atm module:
#                   temperature, pressure, length and density.
#
# 0.11              Added speed conversion
#
# 0.20   06 May 07  Reworked to use default units from default_units module
#
# 0.21   24 Mar 08  Added fuel temperature to avgas_conv
#
# 0.22   25 Apr 08  Corrected error in unit validation in press_conv()
# #############################################################################

import numpy as np

"""
Convert between various units.
"""

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
    default_vol_units = "ft**3"


def area_conv(A, from_units=default_area_units, to_units=default_area_units):
    """
    Convert area values between ft**2, in**2, m**2, km**2, sm**2 and nm**2.

    The incoming value is first converted to ft**2, then it is converted to
    desired return value.


    The units default to those specified in default_units.py

    Examples:

    Convert 1 ft**2 to inches**2, with ft**2 already defined as the default
    units:
    >>> area_conv(1, to_units = 'in**2')
    144.0

    Convert 288 square inches to square feet, with ft**2 already defined as the default
    units:
    >>> area_conv(288, from_units = 'in**2')
    2.0

    Convert 10 square metres to square inches:
    >>> area_conv(1000, from_units = 'm**2', to_units = 'in**2')
    1550003.1000061999
    """

    if from_units == "ft**2":
        pass
    elif from_units == "in**2":
        A /= 144.0
    elif from_units == "m**2":
        A /= 0.3048**2
    elif from_units == "km**2":
        A /= 0.0003048**2
    elif from_units == "sm**2":
        A *= 5280.0**2
    elif from_units == "nm**2":
        A *= (1852 / 0.3048) ** 2
    else:
        raise ValueError(
            'from_units must be "ft**2" or "in**2" or "m**2" or "km**2" or "sm**2" (square statute miles) or "nm**2" (square nautical miles).'
        )

    if to_units == "ft**2":
        return A
    elif to_units == "in**2":
        return A * 144.0
    elif to_units == "m**2":
        return A * 0.3048**2
    elif to_units == "km**2":
        return A * 0.0003048**2
    elif to_units == "sm**2":
        return A / 5280.0**2
    elif to_units == "nm**2":
        return A * (0.3048 / 1852) ** 2
    else:
        raise ValueError(
            'from_units must be "ft**2" or "in**2" or "m**2" or "km**2" or "sm**2" (square statute miles) or "nm**2" (square nautical miles).'
        )


def density_conv(D, from_units, to_units):
    """
    Convert density values between kg/m**3, slug/ft**3 and lb/ft**3.

    The incoming value is first converted to kg/m**3, then it is converted
    to desired return value.

    There are no default units. Both the from_units and the to_units must
    be specified.

    Example:

    Convert 1.225 kg per metre cubed to lb per foot cubed:
    >>> density_conv(1.225, from_units = 'kg/m**3', to_units = 'lb/ft**3')
    0.076474253491112101

    """

    if from_units == "kg/m**3":
        pass
    elif from_units == "slug/ft**3":
        D *= 515.37882
    elif from_units == "lb/ft**3":
        D *= 16.018463
    else:
        raise ValueError(
            'from_units must be one of "kg/m**3", "slug/ft**3" and "lb/ft**3".'
        )

    if to_units == "kg/m**3":
        return D
    elif to_units == "slug/ft**3":
        return D / 515.37882
    elif to_units == "lb/ft**3":
        return D / 16.018463
    else:
        raise ValueError(
            'to_units must be one of "kg/m**3", "slug/ft**3" and "lb/ft**3".'
        )


def force_conv(F, from_units=default_weight_units, to_units=default_weight_units):
    """
    Convert force values between lb and N.

    The incoming value is first converted to N, then it is converted to the
    desired return value.
    """

    if from_units == "N":
        pass
    elif from_units == "lb":
        F *= 4.4482216
    else:
        raise ValueError('from_units must be one of "lb" or "N".')

    if to_units == "N":
        pass
    elif to_units == "lb":
        F /= 4.4482216
    else:
        raise ValueError('to_units must be one of "lb" or "N".')

    return F


def len_conv(L, from_units=default_length_units, to_units=default_length_units):
    """
    Convert length values between ft, in, m, km, sm and nm.

    The incoming value is first converted to ft, then it is converted to
    desired return value.

    The units default to those specified in default_units.py

    Examples:

    Convert 5280 ft to statute miles, with feet already defined as the default
    units:
    >>> len_conv(5280, to_units = 'sm')
    1.0

    Convert 1 nautical mile to feet, with feet already defined as the default
    units:
    >>> len_conv(1, from_units = 'nm')
    6076.1154855643044

    Convert 1000 metres to kilometres:
    >>> len_conv(1000, from_units = 'm', to_units = 'km')
    0.99999999999999989
    """

    if from_units == "ft":
        pass
    elif from_units == "m":
        L /= 0.3048
    elif from_units == "km":
        L /= 0.0003048
    elif from_units == "sm":
        L *= 5280.0
    elif from_units == "nm":
        L *= 1852 / 0.3048
    elif from_units == "in":
        L /= 12.0
    else:
        raise ValueError(
            'from_units must be "ft", "in", "m", "km", "sm" (statute miles) or "nm" (nautical miles).'
        )

    if to_units == "ft":
        return L
    elif to_units == "m":
        return L * 0.3048
    elif to_units == "km":
        return L * 0.0003048
    elif to_units == "sm":
        return L / 5280.0
    elif to_units == "nm":
        return (L * 0.3048) / 1852
    elif to_units == "in":
        return L * 12.0
    else:
        raise ValueError(
            'from_units must be "ft", "in", "m", "km", "sm" (statute miles) or "nm" (nautical miles).'
        )


def power_conv(P, from_units=default_power_units, to_units=default_power_units):
    """
    Convert power values between horsepower, ft-lb/mn,  ft-lb/s, watts,
    kilowatts, BTU/hr and BTU/mn.

    The incoming value is first converted to hp, then it is converted to the
    desired return value.
    The units default to those specified in default_units.py

    """

    if from_units == "hp":
        pass
    elif from_units == "ft-lb/mn":
        P /= 33000.0
    elif from_units == "ft-lb/s":
        P /= 550.0
    elif from_units == "W":
        P /= 745.69987
    elif from_units == "kW":
        P /= 0.74569987
    # elif from_units == 'BTU/hr':
    #     P /= 2544.4332
    # elif from_units == 'BTU/mn':
    #     P /= 42.407227
    else:
        raise ValueError(
            'from_units must be "hp", "ft-lb/mn", "ft-lb/s", "W" (watts), "kW" (kilowatts), "BTU/hr", or "BTU/mn".'
        )

    if to_units == "hp":
        return P
    elif to_units == "ft-lb/mn":
        return P * 33000.0
    elif to_units == "ft-lb/s":
        return P * 550.0
    elif to_units == "W":
        return P * 745.69987
    elif to_units == "kW":
        return P * 0.74569987
    # elif to_units == 'BTU/hr':
    #     return P * 2544.4332
    # elif to_units == 'BTU/mn':
    #     return P * 42.407227
    else:
        raise ValueError(
            'to_units must be "hp", "ft-lb/mn", "ft-lb/s", "W" (watts), "kW" (kilowatts), "BTU/hr", or "BTU/mn".'
        )


@np.vectorize
def press_conv(P, from_units=default_press_units, to_units=default_press_units):
    """
    Convert pressure values between inches of HG, mm of HG, psi, lb/ft^2,
    hpa and mb.

    The incoming value is first converted to Pa, then it is converted to
    desired return value.
    The units default to those specified in default_units.py


    Examples:

    Convert 1013.25 hpa to default pressure units:
    >>> press_conv(1013.25, from_units = 'hpa')
    29.921331923765198

    Convert 29.9213 default pressure units to mm of HG:
    >>> press_conv(29.9213, to_units = 'mm HG')
    760.00128931459176

    Convert 2116.22 lb per sq. ft to lb per sq. inch:
    >>> press_conv(2116.22, from_units = 'psf', to_units = 'psi')
    14.695973160069311
    """

    if from_units == "in HG":
        P *= 3386.38  # from NASA Reference Publication 1046
    elif from_units == "mm HG":
        P *= 133.322  # derived from NASA Reference Publication 1046 value
    elif from_units == "psi":
        P *= 6894.757  # derived from NASA Reference Publication 1046 value
    elif from_units == "psf" or from_units == "lb/ft**2":

        P *= 47.88026  # from NASA Reference Publication 1046
    elif from_units == "hpa" or from_units == "mb":
        P *= 100.0
    elif from_units == "pa":
        pass
    else:
        raise ValueError(
            'from_units must be "in HG", "mm HG", "psi", "psf" (lb per sq. ft), "hpa", "mb" or "pa".'
        )

    if to_units == "in HG":
        return P / 3386.38
    elif to_units == "mm HG":
        return P / 133.322
    elif to_units == "psi":
        return P / 6894.757
    elif to_units == "psf" or to_units == "lb/ft**2":
        return P / 47.88026
    elif to_units == "hpa" or to_units == "mb":
        return P / 100.0
    elif to_units == "pa":
        return P
    else:
        raise ValueError(
            'to_units must be "in HG", "mm HG", "psi", "psf" (lb per sq. ft), "pa", "hpa" or "mb".'
        )


def speed_conv(S, from_units=default_speed_units, to_units=default_speed_units):
    """
    Convert speed values between kt, mph, km/h, m/s and ft/s.

    The incoming value is first converted to kt, then it is converted to
    desired return value.
    The units default to those specified in default_units.py


    Example:

    Convert 230 mph  to kt:
    >>> speed_conv(230, from_units = 'mph', to_units = 'kt')
    199.86453563714903

    """

    if from_units == "kt":
        pass
    elif from_units == "mph":
        S *= len_conv(1, from_units="sm", to_units="nm")
    elif from_units == "km/h":
        S *= len_conv(1, from_units="km", to_units="nm")
    elif from_units == "m/s":
        S *= len_conv(1, from_units="m", to_units="nm") * 3600.0
    elif from_units == "ft/s":
        S *= len_conv(1, from_units=default_length_units, to_units="nm") * 3600.0
    else:
        raise ValueError(
            'from_units must be one of "kt", "mph", "km/h", "m/s" and "ft/s".'
        )

    if to_units == "kt":
        return S
    elif to_units == "mph":
        S *= len_conv(1, from_units="nm", to_units="sm")
        return S
    elif to_units == "km/h":
        S *= len_conv(1, from_units="nm", to_units="km")
        return S
    elif to_units == "m/s":
        S *= len_conv(1, from_units="nm", to_units="m")
        return S / 3600.0
    elif to_units == "ft/s":
        S *= len_conv(1, from_units="nm", to_units=default_length_units)
        return S / 3600.0
    else:
        raise ValueError(
            'to_units must be one of "kt", "mph", "km/h", "m/s" and "ft/s".'
        )


def temp_conv(T, from_units=default_temp_units, to_units=default_temp_units):
    """
    Convert absolute temperature values between deg C, F, K and R.

    This function should not be used for relative temperature conversions,
    i.e. temperature differences.

    The incoming value is first converted to deg K, then it is converted to
    desired return value.
    The units default to those specified in default_units.py


    Examples:

    Convert 32 deg F to deg C, with deg C as the default units:
    >>> temp_conv(32, from_units = 'F')
    0.0

    Convert 100 deg C to deg F, with deg C as the default units:
    >>> temp_conv(100, to_units = 'F')
    212.0

    Convert 59 deg F to deg K
    >>> temp_conv(59, from_units = 'F', to_units = 'K')
    288.14999999999998
    """

    if from_units == "C":
        T += 273.15
    elif from_units == "F":
        T = ((T - 32) * 5.0) / 9.0 + 273.15
    elif from_units == "K":
        pass
    elif from_units == "R":
        T *= 5.0 / 9.0
    else:
        raise ValueError('from_units must be one of "C", "F", "K" or "R".')

    if to_units == "C":
        return T - 273.15
    elif to_units == "F":
        return (T - 273.15) * 1.8 + 32
    elif to_units == "K":
        return T
    elif to_units == "R":
        return T * 1.8
    else:
        raise ValueError('to_units must be one of "C", "F", "K" or "R".')


def vol_conv(V, from_units=default_vol_units, to_units=default_vol_units):
    """
    Convert volume values between USG, ImpGal (Imperial gallons), l (litres), ft**3, in**3, m**3, km**3, sm**3 and nm**3.

    The incoming value is first converted to ft**3, then it is converted to
    desired return value.


    The units default to those specified in default_units.py

    Examples:

    Convert 1 cubic foot to US gallons, with cubic feet already defined as
    the default units:
    >>> vol_conv(1, to_units = 'USG')
    7.4805194804946105

    Convert 1 Imperial gallon to cubic feet, with cubic feet already defined
    as the default units:
    >>> vol_conv(1, from_units = 'ImpGal')
    0.16054365323600001

    Convert 10 US gallon to litres:
    >>> vol_conv(10, from_units = 'USG', to_units = 'l')
    37.854117840125852
    """

    if from_units == "ft**3":
        pass
    elif from_units == "in**3":
        V /= 12.0**3
    elif from_units == "m**3":
        V /= 0.3048**3
    elif from_units == "km**3":
        V /= 0.0003048**3
    elif from_units == "sm**3":
        V *= 5280.0**3
    elif from_units == "nm**3":
        V *= (1852 / 0.3048) ** 3
    elif from_units == "USG":
        V *= 0.133680555556
    elif from_units == "ImpGal":
        V *= 0.160543653236
    elif from_units == "l":
        V /= 3.048**3
    else:
        raise ValueError(
            'from_units must be "ft**3", "in**3", "USG", "ImpGal", "l", "m**3", "km**3", "sm**3" (cubic statute miles) or "nm**3" (cubic nautical miles).'
        )

    if to_units == "ft**3":
        return V
    elif to_units == "in**3":
        return V * 12.0**3
    elif to_units == "m**3":
        return V * 0.3048**3
    elif to_units == "km**3":
        return V * 0.0003048**3
    elif to_units == "sm**3":
        return V / 5280.0**3
    elif to_units == "nm**3":
        return V * (0.3048 / 1852) ** 3
    elif to_units == "USG":
        return V / 0.133680555556
    elif to_units == "ImpGal":
        return V / 0.160543653236
    elif to_units == "l":
        return V * 3.048**3
    else:
        raise ValueError(
            'to_units must be "ft**3", "in**3", "USG", "ImpGal", "l", "m**3", "km**3", "sm**3" (cubic statute miles) or "nm**3" (cubic nautical miles).'
        )


def wt_conv(W, from_units=default_weight_units, to_units=default_weight_units):
    """
    Convert weight values between lb and kg.

    Purists will yell that lb is a unit of weight, and kg is a unit of mass.
    Get over it.

    The incoming value is first converted to kg, then it is converted to the
    desired return value.

    The units default to those specified in default_units.py


    """

    if from_units == "kg":
        pass
    elif from_units == "lb":
        W *= 0.453592
    else:
        raise ValueError('from_units must be one of "lb" or "kg".')

    if to_units == "kg":
        pass
    elif to_units == "lb":
        W *= 2.204622622
    else:
        raise ValueError('to_units must be one of "lb" or "kg".')

    return W


def avgas_conv(
    AG,
    from_units=default_avgas_units,
    to_units=default_avgas_units,
    temp=15,
    temp_units="C",
    grade="nominal",
):
    """
    Convert aviation gasoline between units of lb, US Gallon (USG),
    Imperial Gallon (Imp Gal), litres (l) and kg, assuming nominal
    density for aviation gasoline of 6.01 lb per USG.

    The units default to those specified in default_units.py

    Note: it was difficult to find authoritative values for aviation gasoline
    density.  Conventional wisdom is that aviation gasoline has a density of
    6 lb/USG.  The Canada Flight Supplement provides densities of:
    temp      density     density    density
    (deg C)   (lb/USG)  (lb/ImpGal)  (lb/l)
    -40         6.41       7.68       1.69
    -20         6.26       7.50       1.65
      0         6.12       7.33       1.62
     15         6.01       7.20       1.59
     30         5.90       7.07       1.56

    However, the Canada Flight Supplement does not provide a source for its
    density data.  And, the values for the different volume units are not
    completly consistent, as they don't vary by exactly the correct factor.
    For example, if the density at 15 deg C is 6.01 lb/USG, we would expect
    the density in lb/ImpGal to be 7.22, (given that 1 ImpGal = 1.201 USG)
    yet the Canada Flight Supplement has 7.20.

    The only authoritative source for aviation gasoline density that was
    found on the web was the \"Air BP Handbook of Products\" on the British
    Petroleum (BP) web site:

    <http://www.bp.com/liveassets/bp_internet/aviation/air_bp/STAGING/local_assets/downloads_pdfs/a/air_bp_products_handbook_04004_1.pdf>

    It provides the following density data valid at 15 deg C (the BP document
    only provides density in kg/m**3 - the density in lb/USG were calculated
    by Kevin Horton):
    Avgas    density     density
    Type    (kg/m**3)    (lb/USG)
    80       690          5.76
    100      695          5.80
    100LL    715          5.97

    The available aviation gasoline specifications do not appear to define an
    allowable density range.  They do define allowable ranges for various
    parametres of the distillation process - the density of the final product
    will vary depending on where in the allowable range the refinery is run.
    Thus there will be some variation in density from refinery to refinery.

    This function uses the 15 deg C density values provided by BP, with the
    variation with temperature provided in the Canada Flight Supplement.

    The grade may be specified as \"80\", \"100\" or \"100LL\".  It defaults to
    \"100LL\" if it is not specified.

    The temperature defaults to 15 deg C if it is not specified.
    """

    lb_per_USG_15_nom = (
        6.01  # nominal density at 15 deg C from Canada Flight Supplement
    )
    slope = (
        -0.007256
    )  # change in density per deg C based on data from Canada Flight Supplement

    lb_per_USG = lb_per_USG_15_nom * (
        1
        + (slope * (temp_conv(temp, from_units=temp_units, to_units="C") - 15))
        / lb_per_USG_15_nom
    )  # density corrected for temperature, using nominal density value

    if grade == "nominal":
        grade_density = lb_per_USG_15_nom
    elif grade == "100LL":
        grade_density = 5.967
    elif str(grade) == "100":
        grade_density = 5.801
    elif str(grade) == "80":
        grade_density = 5.7583
    else:
        raise ValueError(
            'grade must be one of "nominal", "80", "100" or "100LL", with a default of "100LL"'
        )

    # Correct the density if the grade is other than nominal.
    # If the grade actually is nominal, we are multiplying by 1 / 1

    lb_per_USG *= grade_density / lb_per_USG_15_nom

    if from_units == "lb":
        pass
    elif from_units == "USG":
        AG *= lb_per_USG
    elif from_units == "ImpGal":
        AG *= vol_conv(lb_per_USG, from_units="ImpGal", to_units="USG")
    elif from_units == "kg":
        AG = wt_conv(AG, from_units="kg")
    elif from_units == "l":
        AG *= vol_conv(lb_per_USG, from_units="l", to_units="USG")
    else:
        raise ValueError(
            'from_units must be one of "lb", "USG", "Imp Gal", "l", or "kg".'
        )

    if to_units == "lb":
        pass
    elif to_units == "USG":
        AG /= lb_per_USG
    elif to_units == "ImpGal":
        AG /= vol_conv(lb_per_USG, from_units="ImpGal", to_units="USG")
    elif to_units == "kg":
        AG = wt_conv(AG, to_units="kg")
    elif to_units == "l":
        AG /= vol_conv(lb_per_USG, from_units="l", to_units="USG")
    else:
        raise ValueError(
            'from_units must be one of "lb", "USG", "Imp Gal", "l", or "kg".'
        )

    return AG
