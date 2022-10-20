# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
from datetime import datetime, timedelta
from math import asin, atan2, cos, degrees, radians, sin, sqrt
from tempfile import NamedTemporaryFile
from typing import Callable, Collection, Iterable, Tuple, Union

import cdsapi
import cfgrib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cartopy import crs as ccrs
from cartopy.feature import BORDERS, LAND, OCEAN
from matplotlib.figure import Figure
from openap import aero, nav

# from openap.extra.aero import ft, h_isa
# from openap.top import wind
from scipy.interpolate import RegularGridInterpolator


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print(
                "[%s]" % self.name,
            )
        print("Elapsed: %s" % (time.time() - self.tstart))


class WeatherRetrieverFromEcmwf:
    def __init__(
        self,
        url: str = "https://cds.climate.copernicus.eu/api/v2",
        key: str = None,
        pressure_levels: Iterable[str] = [
            "150",
            "175",
            "200",
            "225",
            "250",
            "300",
            "350",
            "400",
            "450",
            "500",
            "550",
            "600",
            "650",
            "700",
            "750",
            "775",
            "800",
            "825",
            "850",
            "875",
            "900",
            "925",
            "950",
            "975",
            "1000",
        ],
        times: Iterable[str] = [
            "00:00",
            "01:00",
            "02:00",
            "03:00",
            "04:00",
            "05:00",
            "06:00",
            "07:00",
            "08:00",
            "09:00",
            "10:00",
            "11:00",
            "12:00",
            "13:00",
            "14:00",
            "15:00",
            "16:00",
            "17:00",
            "18:00",
            "19:00",
            "20:00",
            "21:00",
            "22:00",
            "23:00",
        ],
    ):
        """Wind data data retriever from ECMWF

        Args:
            url (str, optional): the URL of the service. Defaults to "https://cds.climate.copernicus.eu/api/v2".
            key (str, optional): the API key. Defaults to None.
            pressure_levels (Iterable[str], optional): The pressure levels. Defaults to [ "150", "175", "200", "225", "250", "300", "350", "400", "450", "500", "550", "600", "650", "700", "750", "775", "800", "825", "850", "875", "900", "925", "950", "975", "1000", ].
            times (Iterable[str], optional): The times. Defaults to [ "00:00", "01:00", "02:00", "03:00", "04:00", "05:00", "06:00", "07:00", "08:00", "09:00", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00", "17:00", "18:00", "19:00", "20:00", "21:00", "22:00", "23:00", ].
        """
        self.c = cdsapi.Client(url=url, key=key)
        self.pressure_levels = pressure_levels
        self.times = times

    def get(self, dt: datetime):
        """Get the weather data for a given datetime

        Args:
            dt (datetime): the datetime

        Returns:
            _type_: _description_
        """
        file = NamedTemporaryFile(suffix=".grib")

        self.c.retrieve(
            "reanalysis-era5-pressure-levels",
            {
                "product_type": "reanalysis",
                "variable": [
                    "u_component_of_wind",
                    "v_component_of_wind",
                ],
                "pressure_level": self.pressure_levels,
                "year": dt.year,
                "month": dt.month,
                "day": dt.day,
                "time": self.times,
                "format": "grib",
            },
            f"{file.name}",
        )

        return file


class WindInterpolator:
    def __init__(self, path: str):
        """The wind interpolator class

        Args:
            path (str): path to the wind file
        """
        self.dataset = cfgrib.open_dataset(
            path,
            backend_kwargs={
                "filter_by_keys": {"typeOfLevel": "isobaricInhPa"},
                "indexpath": "",
            },
        )

        # latitude: [90, -90]
        # longitude: [0, 360[
        # isobaricInhPa: [1000, 150]

    def get_dataset(self) -> xr.Dataset:
        """Get the dataset

        Returns:
            xr.Dataset: the dataset
        """
        return self.dataset

    def get_windfield(self, ts: float) -> pd.DataFrame:
        """Get the wind field at a given time

        Args:
            ts (float): the time in seconds from the start of the dataset

        Returns:
            pd.DataFrame: a dataframe with the wind field
        """
        time = self.dataset.time[0] + np.timedelta64(int(ts), "s")
        with Timer("Get Windfield"):
            df = (
                self.dataset.interp(time=time)
                .to_dataframe()
                # .drop(
                #    ["isobaricInhPa", "number", "time", "step", "valid_time"],
                #    axis=1,
                # )
                .reset_index()
                .assign(longitude=lambda d: (d.longitude + 180) % 360 - 180)
                .assign(h=lambda d: aero.h_isa(d.isobaricInhPa * 100))
            )
        return df

    def plot_wind(
        self,
        alt: float = 35000.0,
        t: Union[float, Iterable[float]] = 0.0,
        plot_wind: bool = False,
        plot_barbs: bool = False,
        ax: plt.axes = None,
    ) -> plt.axes:
        """_summary_

        Args:
            alt (float, optional): altitude layer to plot. Defaults to 35000.0.
            t (Union[float, Iterable[float]], optional): value of time step (in second) or list/array of time step (in s). Defaults to 0.0.
            plot_wind (bool, optional): plot the wind vector field. Defaults to False.
            plot_barbs (bool, optional): plot the barbs. Defaults to False.
            ax (plt.axes, optional): Axes object where to plot the wind. Defaults to None.

        Returns:
            plt.axes: The axes with the plot
        """
        if isinstance(t, Collection):
            times = t
        else:
            times = [t]

        if ax is None:
            fig, ax = plt.subplots(
                1, 1, figsize=(10, 5), subplot_kw={"projection": ccrs.PlateCarree()}
            )
            ax.stock_img()

        p, _, _ = aero.atmos(alt * aero.ft)

        for t in times:
            with Timer("Interpolation"):
                ds = self.dataset.interp(
                    time=self.dataset.time[0] + np.timedelta64(t, "s"),
                    isobaricInhPa=p / 100,
                    latitude=self.dataset.latitude.values[::10],
                    longitude=self.dataset.longitude.values[::10],
                )

                Ut = ds.u
                Vt = ds.v
                x, y = np.meshgrid(ds.longitude.values, ds.latitude.values)

            Nt = np.sqrt(np.square(Ut) + np.square(Vt))

            with Timer("Plot contour"):
                cs = ax.contourf(x, y, Nt, 100, alpha=0.5, zorder=2)
            with Timer("Plot wind"):
                if plot_wind:
                    if x.shape[0] > 100:
                        q = ax.quiver(
                            x[::10, ::10],
                            y[::10, ::10],
                            Ut[::10, ::10],
                            Vt[::10, ::10],
                            alpha=0.8,
                        )
                    else:
                        q = ax.quiver(x, y, Ut[:, :], Vt[:, :], alpha=0.8)

            if plot_barbs:
                ax.barbs(
                    x,
                    y,
                    Ut,
                    Vt,
                    transform=ccrs.PlateCarree(),
                    color="k",
                    length=5,
                    lw=0.5,
                    label=f"Wind FL{int(alt/100)}",
                )

            plt.title(f"time : {t} second(s)")
            # plt.draw()
            # plt.pause(0.1)

            # for c in cs.collections:
            #     plt.gca().collections.remove(c)

            # if plot_wind:
            #     q.remove()

        return ax


def flying(
    from_: pd.DataFrame, to_: Tuple[float, float], ds: xr.Dataset, fflow: Callable
) -> pd.DataFrame:
    """Compute the trajectory of a flying object from a given point to a given point

    Args:
        from_ (pd.DataFrame): the trajectory of the object so far
        to_ (Tuple[float, float]): the destination of the object
        ds (xr.Dataset): dataset containing the wind field
        fflow (Callable): fuel flow function

    Returns:
        pd.DataFrame: the final trajectory of the object
    """
    pos = from_.to_dict("records")[0]

    dist_ = aero.distance(pos["lat"], pos["lon"], to_[0], to_[1], pos["alt"])
    data = []

    epsilon = 1_000
    dt = 600

    dist = dist_

    loop = 0
    while dist > epsilon:  # or loop < 20 or dt > 0:
        bearing = aero.bearing(pos["lat"], pos["lon"], to_[0], to_[1])

        p, _, _ = aero.atmos(pos["alt"] * aero.ft)
        isobaric = p / 100

        we, wn = 0, 0
        if ds:
            time = ds.time[0] + np.timedelta64(pos["ts"], "s")

            wind_ms = ds.interp(
                latitude=pos["lat"],
                longitude=pos["lon"],
                time=time,
                isobaricInhPa=isobaric,
            )
            we, wn = wind_ms.u.values, wind_ms.v.values  # 0, 300

        wdir = (degrees(atan2(we, wn)) + 180) % 360
        wspd = sqrt(wn * wn + we * we)

        tas = aero.mach2tas(pos["mach"], pos["alt"])  # 400

        wca = asin((wspd / tas) * sin(radians(bearing - wdir)))

        # ground_speed = sqrt(
        #     tas * tas
        #     + wspd * wspd
        #     + 2 * tas * wspd * cos(radians(bearing - wdir - wca))
        # )

        heading = (360 + bearing - degrees(wca)) % 360

        gsn = tas * cos(radians(heading)) - wn
        gse = tas * sin(radians(heading)) - we

        gs = sqrt(gsn * gsn + gse * gse)

        brg = degrees(atan2(gse, gsn)) % 360.0

        ll = aero.latlon(pos["lat"], pos["lon"], gs * dt, brg, pos["alt"])
        pos["fuel"] = dt * fflow(
            pos["mass"], tas / aero.kts, pos["alt"] * aero.ft, path_angle=0.0
        )
        mass = pos["mass"] - pos["fuel"]

        new_row = {
            "ts": pos["ts"] + dt,
            "lat": ll[0],
            "lon": ll[1],
            "mass": mass,
            "mach": pos["mach"],
            "fuel": pos["fuel"],
            "alt": pos["alt"],
        }

        dist = aero.distance(
            new_row["lat"], new_row["lon"], to_[0], to_[1], new_row["alt"]
        )

        if dist < dist_:
            data.append(new_row)
            dist_ = dist

            pos = data[-1]
        else:
            dt = int(dt / 10)
            assert dt > 0

        loop += 1

    return pd.DataFrame(data)


def plot_trajectory(
    lat1, lon1, lat2, lon2, trajectory: pd.DataFrame, ds: xr.Dataset
) -> Figure:
    """Plot the trajectory of an object

    Args:
        trajectory (pd.DataFrame): the trajectory of the object
        ds (xr.Dataset): the dataset containing the wind field

    Returns:
        Figure: the figure
    """

    fig = Figure(figsize=(600, 600))
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False
    fig.canvas.resizable = False
    fig.set_dpi(1)

    # lon1, lat1 = trajectory.iloc[0]["lon"], trajectory.iloc[0]["lat"]
    # lon2, lat2 = trajectory.iloc[-1]["lon"], trajectory.iloc[-1]["lat"]

    latmin, latmax = min(lat1, lat2), max(lat1, lat2)
    lonmin, lonmax = min(lon1, lon2), max(lon1, lon2)

    ax = plt.axes(
        projection=ccrs.TransverseMercator(
            central_longitude=(lonmax - lonmin) / 2,
            central_latitude=(latmax - latmin) / 2,
        )
    )

    wind_sample = 30

    ax.set_extent([lonmin - 4, lonmax + 4, latmin - 2, latmax + 2])
    ax.add_feature(OCEAN, facecolor="#d1e0e0", zorder=-1, lw=0)
    ax.add_feature(LAND, facecolor="#f5f5f5", lw=0)
    ax.add_feature(BORDERS, lw=0.5, color="gray")
    ax.gridlines(draw_labels=True, color="gray", alpha=0.5, ls="--")
    ax.coastlines(resolution="50m", lw=0.5, color="gray")

    if ds is not None:
        t = trajectory.ts.iloc[-1]
        alt = trajectory.alt.iloc[-1]
        """
        self.wind_interpolator.plot_wind(
            alt=alt,
            lon_min=max(-180, lonmin - 4),
            lon_max=min(+180, lonmax + 4),
            lat_min=max(-90, latmin - 2),
            lat_max=min(+90, latmax + 2),
            t=int(t),
            n_lat=180,
            n_lon=720,
            plot_wind=False,
            plot_barbs=False,
            ax=ax,
        )
        """

    # great circle
    ax.scatter(lon1, lat1, c="darkgreen", transform=ccrs.Geodetic())
    ax.scatter(lon2, lat2, c="tab:red", transform=ccrs.Geodetic())

    ax.plot(
        [lon1, lon2],
        [lat1, lat2],
        label="Great Circle",
        color="tab:red",
        ls="--",
        transform=ccrs.Geodetic(),
    )

    # trajectory
    ax.plot(
        trajectory.lon,
        trajectory.lat,
        color="tab:green",
        transform=ccrs.Geodetic(),
        linewidth=2,
        marker=".",
        label="Optimal",
    )

    ax.legend()

    # Save it to a temporary buffer.
    # buf = BytesIO()
    # fig.savefig(buf, format="png")
    # Embed the result in the html output.
    # data = base64.b64encode(buf.getbuffer()).decode("ascii")

    return fig
