# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import time
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cartopy import crs as ccrs
from cartopy.feature import BORDERS, LAND, OCEAN
from matplotlib.figure import Figure
from openap import aero
from pygeodesy.ellipsoidalVincenty import LatLon


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


def plot_trajectory(lat1, lon1, lat2, lon2, trajectory: pd.DataFrame) -> Figure:
    """Plot the trajectory of an object

    Args:
        trajectory (pd.DataFrame): the trajectory of the object

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

    ax = plt.axes(projection=ccrs.TransverseMercator())

    ax.set_extent([lonmin - 4, lonmax + 4, latmin - 2, latmax + 2])
    ax.add_feature(OCEAN, facecolor="#d1e0e0", zorder=-1, lw=0)
    ax.add_feature(LAND, facecolor="#f5f5f5", lw=0)
    ax.add_feature(BORDERS, lw=0.5, color="gray")
    ax.gridlines(draw_labels=True, color="gray", alpha=0.5, ls="--")
    ax.coastlines(resolution="50m", lw=0.5, color="gray")

    # great circle
    ax.scatter(lon1, lat1, c="darkgreen", transform=ccrs.Geodetic())
    ax.scatter(lon2, lat2, c="red", transform=ccrs.Geodetic())

    ax.plot(
        [lon1, lon2],
        [lat1, lat2],
        label="Great Circle",
        color="red",
        ls="--",
        transform=ccrs.Geodetic(),
    )

    # trajectory
    ax.plot(
        trajectory.lon,
        trajectory.lat,
        color="green",
        transform=ccrs.Geodetic(),
        linewidth=2,
        marker=".",
        label="Optimal",
    )

    ax.legend()

    return fig


def plot_trajectory_no_map(lat1, lon1, lat2, lon2, trajectory: pd.DataFrame) -> Figure:
    """Plot the trajectory of an object

    Args:
        trajectory (pd.DataFrame): the trajectory of the object

    Returns:
        Figure: the figure
    """

    fig = Figure(figsize=(600, 600))
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False
    fig.canvas.resizable = False
    fig.set_dpi(1)

    latmin, latmax = min(lat1, lat2), max(lat1, lat2)
    lonmin, lonmax = min(lon1, lon2), max(lon1, lon2)

    fig, ax = plt.subplots(1)

    wind_sample = 30
    ax.set_xlim([lonmin, lonmax])
    ax.set_ylim([latmin, latmax])
    # great circle
    ax.scatter(lon1, lat1, c="darkgreen")
    ax.scatter(lon2, lat2, c="red")

    ax.plot(
        [lon1, lon2],
        [lat1, lat2],
        label="Great Circle",
        color="red",
        ls="--",
    )

    # trajectory
    ax.plot(
        trajectory.lon,
        trajectory.lat,
        color="green",
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


def plot_altitude(trajectory: pd.DataFrame) -> Figure:
    fig = plt.Figure()
    ax = plt.axes()
    pos = [
        LatLon(
            trajectory.iloc[i]["lat"],
            trajectory.iloc[i]["lon"],
            trajectory.iloc[i]["alt"],
        )
        for i in range(len(trajectory.alt))
    ]
    dist = [d.distanceTo(pos[0]) for d in pos]
    ax.plot(dist, trajectory.alt)
    return fig


def plot_network(domain, dir=None):
    network = domain.network
    origin_coord = domain.lat1, domain.lon1, domain.alt1
    target_coord = domain.lat2, domain.lon2, domain.alt2
    fig, ax = plt.subplots(1, subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_extent(
        [
            min(origin_coord[1], target_coord[1]) - 4,
            max(origin_coord[1], target_coord[1]) + 4,
            min(origin_coord[0], target_coord[0]) - 2,
            max(origin_coord[0], target_coord[0]) + 2,
        ]
    )
    ax.add_feature(OCEAN, facecolor="#d1e0e0", zorder=-1, lw=0)
    ax.add_feature(LAND, facecolor="#f5f5f5", lw=0)
    ax.add_feature(BORDERS, lw=0.5, color="gray")
    ax.gridlines(draw_labels=True, color="gray", alpha=0.5, ls="--")
    ax.coastlines(resolution="50m", lw=0.5, color="gray")
    ax.scatter(
        [
            network[x][x1][x2].lon
            for x in range(len(network))
            for x1 in range(len(network[x]))
            for x2 in range(len(network[x][x1]))
        ],
        [
            network[x][x1][x2].lat
            for x in range(len(network))
            for x1 in range(len(network[x]))
            for x2 in range(len(network[x][x1]))
        ],
        transform=ccrs.Geodetic(),
        s=0.2,
    )

    if dir:
        fig.savefig(f"{dir}/network points.png")
    else:
        fig.savefig("network points.png")


def trajectory_on_map(df, windfield=None, ax=None, wind_sample=4):

    lat1, lon1 = df.lat.iloc[0], df.lon.iloc[0]
    lat2, lon2 = df.lat.iloc[-1], df.lon.iloc[-1]

    latmin, latmax = min(lat1, lat2), max(lat1, lat2)
    lonmin, lonmax = min(lon1, lon2), max(lon1, lon2)

    if ax is None:
        ax = plt.axes(
            projection=ccrs.TransverseMercator(
                central_longitude=df.lon.mean(), central_latitude=df.lat.mean()
            )
        )

    ax.set_extent([lonmin - 4, lonmax + 4, latmin - 2, latmax + 2])
    ax.add_feature(OCEAN, facecolor="#d1e0e0", zorder=-1, lw=0)
    ax.add_feature(LAND, facecolor="#f5f5f5", lw=0)
    ax.add_feature(BORDERS, lw=0.5, color="gray")
    ax.gridlines(draw_labels=True, color="gray", alpha=0.5, ls="--")
    ax.coastlines(resolution="50m", lw=0.5, color="gray")

    if windfield is not None:
        # get the closed altitude
        h_max = df.alt.max() * aero.ft
        fl = int(round(h_max / aero.ft / 100, -1))
        idx = np.argmin(abs(windfield.h.unique() - h_max))
        df_wind = (
            windfield.query(f"h=={windfield.h.unique()[idx]}")
            .query(f"longitude <= {lonmax + 2}")
            .query(f"longitude >= {lonmin - 2}")
            .query(f"latitude <= {latmax + 2}")
            .query(f"latitude >= {latmin - 2}")
        )

        ax.barbs(
            df_wind.longitude.values[::wind_sample],
            df_wind.latitude.values[::wind_sample],
            df_wind.u.values[::wind_sample],
            df_wind.v.values[::wind_sample],
            transform=ccrs.PlateCarree(),
            color="k",
            length=5,
            lw=0.5,
            label=f"Wind FL{fl}",
        )

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
        df.lon,
        df.lat,
        color="tab:green",
        transform=ccrs.Geodetic(),
        linewidth=2,
        marker=".",
        label="Optimal",
    )

    ax.legend()

    return plt
