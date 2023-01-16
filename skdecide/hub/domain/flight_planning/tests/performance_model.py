import sys

import numpy as np

sys.path.append("../")
from math import asin, atan2, cos, degrees, radians, sin, sqrt
from typing import Callable

import pandas as pd
from openap.extra.aero import atmos
from openap.extra.aero import bearing as aero_bearing
from openap.extra.aero import distance, ft, kts, latlon, mach2tas
from openap.extra.nav import airport
from openap.fuel import FuelFlow
from openap.prop import aircraft
from pygeodesy.ellipsoidalVincenty import LatLon


def load_needed():
    actype = "A320"
    ac = aircraft(actype)

    ac["limits"]["ceiling"] /= ft
    ac["cruise"]["height"] /= ft
    mach = ac["cruise"]["mach"]
    lat1, lon1 = 40, 1
    dataf = pd.DataFrame(
        [
            {
                "ts": 0,
                "lat": lat1,
                "lon": lon1,
                "mass": ac["limits"]["MTOW"]
                - 500.0,  # Here we compute the weight difference between the fuel loaded and the fuel capacity
                "mach": mach,
                "fuel": 0.0,
                "alt": 35000,
            }
        ]
    )
    lat2, lon2, alt2 = 50, 1, 35000
    fuel_flow = FuelFlow(actype).enroute
    perf_model(
        from_=dataf,
        to_=(lat2, lon2, alt2),
        wind_value=(0, -10),
        mach=mach,
        fuel_flow=fuel_flow,
    )


class Wind:
    def __init__(self):
        self.error = ""
        self.gs = 0
        self.wca = 0

    ## Formula to calculate correction for wind drift
    ## Inputs:
    ## TAS:           True airspeed in kts
    ## WindSpeed:     Wind speed in kts
    ## WindDirection: Wind direction in radian  degree
    ## TC:            True course in radian :  deg
    ## Outputs:
    ## self.WCA:      Wind correction angle in radian degree
    ## self.GS:          Ground speed in kts
    ## self.Error:    Error message, empty string: no error

    def CalcCorrection(self, TAS, TC, WindSpeed, WindDirection):
        self.tas = float(TAS)
        self.ws = float(WindSpeed)
        self.wd = float(WindDirection)  # * pi / 180.0
        self.tc = float(TC)  # * pi / 180.0

        # calculate wind correction angle wca and ground speed gs
        self.swc = (self.ws / self.tas) * sin(self.wd - self.tc)
        if abs(self.swc) >= 1.0:
            # Wind is to strong
            self.error = "Wind is too strong"
        else:
            self.wca = asin(self.swc)  # * 180.0 / pi)
            self.gs = self.tas * sqrt(1 - self.swc * self.swc) - self.ws * cos(
                self.wd - self.tc
            )

        if self.gs < 0:
            # Wind is to strong
            self.gs = self.tas
            self.error = "Wind is too strong"
        else:
            # Reset possible status message
            self.error = ""

    def CalcTAS(self, GS, TC, WindSpeed, WindDirection):
        return GS + WindSpeed * cos(WindDirection - TC)

    def Error(self):
        return self.error

    def WCA(self):
        return self.wca

    def GS(self):
        return self.gs


def perf_model(
    from_,
    to_,
    fuel_flow: Callable,
    wind_interpolator=None,
    wind_value=None,
    mach: float = 0.8,
):
    pos = from_.to_dict("records")[0]
    alt = to_[2]
    dist_ = distance(pos["lat"], pos["lon"], to_[0], to_[1], alt)
    data = []
    epsilon = 100
    dt = 600
    dist = dist_
    loop = 0
    while dist > epsilon:
        bearing = aero_bearing(pos["lat"], pos["lon"], to_[0], to_[1])
        print("Bearing ", bearing)
        print("Cur pos :", pos["lat"], pos["lon"])
        p, _, _ = atmos(alt * ft)
        isobaric = p / 100
        we, wn = 0, 0
        if wind_interpolator:
            time = pos["ts"]
            wind_ms = wind_interpolator.interpol_wind_classic(
                lat=pos["lat"], longi=pos["lon"], alt=alt, t=time
            )
            we, wn = wind_ms[2][0], wind_ms[2][1]  # 0, 300
        else:
            we, wn = wind_value

        angle = atan2(wn, we)
        angle = 3.0 * np.pi / 2.0 - angle

        wdir = (degrees(atan2(we, wn)) + 180) % 360
        wspd = sqrt(wn * wn + we * we)
        wspd = sqrt(wn * wn + we * we)

        tas = mach2tas(mach, alt * ft)  # 400

        wi = Wind()
        wi.CalcCorrection(TAS=tas, TC=bearing, WindSpeed=wspd, WindDirection=angle)
        gs = wi.gs
        print("ground speed method1 : ", gs)

        print("tas = ", tas)
        if False:
            wca = asin((wspd / tas) * sin(radians(bearing - wdir)))
            heading = (360 + bearing - degrees(wca)) % 360
            gsn = tas * cos(radians(heading)) - wn
            gse = tas * sin(radians(heading)) - we
            gs = sqrt(gsn * gsn + gse * gse)  # ground speed
        print("Ground speed method 2", gs)
        if gs * dt > dist:
            # Last step. make sure we go to destination.
            dt = dist / gs
            ll = to_[0], to_[1]
        else:
            # brg = degrees(atan2(gse, gsn)) % 360.0
            ll = latlon(pos["lat"], pos["lon"], gs * dt, bearing, alt * ft)

        pos["fuel"] = dt * fuel_flow(
            pos["mass"],
            tas / kts,
            alt * ft,
            path_angle=(alt - pos["alt"]) / (gs * dt),
        )
        mass = pos["mass"] - pos["fuel"]

        new_row = {
            "ts": pos["ts"] + dt,
            "lat": ll[0],
            "lon": ll[1],
            "mass": mass,
            "mach": mach,
            "fuel": pos["fuel"],
            "alt": alt,  # to be modified
        }

        # New distance to the next 'checkpoint'
        dist = distance(new_row["lat"], new_row["lon"], to_[0], to_[1], new_row["alt"])

        # print("Dist : %f Dist_ : %f " %(dist,dist_))
        if dist < dist_:
            # print("Fuel new_row : %f" %new_row["fuel"])
            data.append(new_row)
            dist_ = dist
            pos = data[-1]
        else:
            dt = int(dt / 10)
            print("going in the wrong part.")
            assert dt > 0

        loop += 1
    print("Fuel burnt : ", data[0]["mass"] - data[-1]["mass"])
    print("Time : ", data[-1]["ts"] - data[0]["ts"])

    return pd.DataFrame(data)


if __name__ == "__main__":
    load_needed()
