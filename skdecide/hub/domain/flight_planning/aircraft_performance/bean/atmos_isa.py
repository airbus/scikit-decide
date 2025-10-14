from openap.extra.aero import ft

TK = 273.15


def temperature(altitude_ft, disa=0.0):
    # Setup the dictionnary containing constants depending on zones
    temp_constant = {}
    temp_constant["zone1"] = {"A": 288.15, "B": -6.5e-3}

    temp_constant["zone2"] = {"A": 216.65, "B": 0.0}

    temp_constant["zone3"] = {"A": 196.65, "B": 1e-3}

    temp_constant["zone4"] = {"A": 139.05, "B": 2.8e-3}

    temp_constant["zone5"] = {"A": 270.65, "B": 0.0}

    altitude_m = altitude_ft * ft

    if altitude_m <= 11000:
        zone = "zone1"
        temperature_k = (
            temp_constant[zone]["A"] + temp_constant[zone]["B"] * altitude_m + disa
        )
    elif altitude_m <= 20000:
        zone = "zone2"
        temperature_k = (
            temp_constant[zone]["A"] + temp_constant[zone]["B"] * altitude_m + disa
        )
    elif altitude_m <= 32000:
        zone = "zone3"
        temperature_k = (
            temp_constant[zone]["A"] + temp_constant[zone]["B"] * altitude_m + disa
        )

    elif altitude_m <= 47000:
        zone = "zone4"
        temperature_k = (
            temp_constant[zone]["A"] + temp_constant[zone]["B"] * altitude_m + disa
        )

    else:
        zone = "zone5"
        temperature_k = (
            temp_constant[zone]["A"] + temp_constant[zone]["B"] * altitude_m + disa
        )

    return temperature_k


def disa_alt_temp(altitude_ft: float, temperature_K: float = 288.15):
    disa = temperature_K - temperature(altitude_ft, disa=0.0)

    return disa
