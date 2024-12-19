# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import time
from copy import deepcopy

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cartopy import crs as ccrs
from cartopy.feature import BORDERS, LAND, OCEAN
from matplotlib.figure import Figure
from openap import aero
from openap.extra.aero import ft, nm
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


def plot_full(domain, trajectory: pd.DataFrame) -> Figure:
    network = domain.network

    fig = plt.figure(figsize=(10, 7))

    # define the grid layout
    gs = gridspec.GridSpec(3, 2, width_ratios=[2, 4])

    # add subplots for the line plots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])

    # values
    pos = [
        LatLon(
            trajectory.iloc[i]["lat"],
            trajectory.iloc[i]["lon"],
            trajectory.iloc[i]["alt"] * ft,
        )
        for i in range(len(trajectory.alt))
    ]
    dist = np.array([d.distanceTo(pos[0]) for d in pos])

    # plot the altitude
    ax1.plot(dist / nm, trajectory.alt)
    ax1.set_xlabel("ESAD (nm)")
    ax1.set_ylabel("Zp (ft)")
    ax1.set_title("Altitude profile")

    # plot the mass
    ax2.plot(dist / nm, trajectory.mass)
    ax2.set_xlabel("ESAD (nm)")
    ax2.set_ylabel("Mass (Kg)")
    ax2.set_title("Mass profile")

    # create twin axis for ax3
    ax3_twin = ax3.twinx()
    # plot CAS / mach profile with altitude on y axis
    ax3.plot(dist / nm, trajectory.cas, label="CAS", color="blue")
    ax3_twin.plot(dist / nm, trajectory.mach, label="Mach", color="red")
    ax3.set_xlabel("ESAD (nm)")
    ax3.set_ylabel("CAS (kt)")
    ax3_twin.set_ylabel("Mach")
    ax3.set_title("CAS / Mach profile")
    # add legend
    ax3.legend(loc="upper left")
    ax3_twin.legend(loc="upper right")

    # plot the trajectory
    latmin, latmax = min(trajectory.lat), max(trajectory.lat)
    lonmin, lonmax = min(trajectory.lon), max(trajectory.lon)

    ax4 = fig.add_subplot(
        gs[:, 1], projection=ccrs.PlateCarree(central_longitude=trajectory.lon.mean())
    )

    ax4.set_extent([lonmin - 4, lonmax + 4, latmin - 2, latmax + 2])
    ax4.add_feature(OCEAN, facecolor="#d1e0e0", zorder=-1, lw=0)
    ax4.add_feature(LAND, facecolor="#f5f5f5", lw=0)
    ax4.add_feature(BORDERS, lw=0.5, color="gray")
    ax4.gridlines(draw_labels=True, color="gray", alpha=0.5, ls="--")
    ax4.coastlines(resolution="50m", lw=0.5, color="gray")

    # add great circle
    ax4.scatter(
        trajectory.lon.iloc[0],
        trajectory.lat.iloc[0],
        c="darkgreen",
        transform=ccrs.Geodetic(),
    )
    ax4.scatter(
        trajectory.lon.iloc[-1],
        trajectory.lat.iloc[-1],
        c="red",
        transform=ccrs.Geodetic(),
    )

    ax4.plot(
        [trajectory.lon.iloc[0], trajectory.lon.iloc[-1]],
        [trajectory.lat.iloc[0], trajectory.lat.iloc[-1]],
        label="Great Circle",
        color="red",
        ls="--",
        transform=ccrs.Geodetic(),
    )

    # add trajectory
    ax4.plot(
        trajectory.lon,
        trajectory.lat,
        color="green",
        transform=ccrs.Geodetic(),
        linewidth=2,
        marker=".",
        label="skdecide",
    )
    ax4.legend()
    total_fuel = np.round(trajectory["fuel"].sum(), 2)
    total_time = trajectory["ts"].values[-1] - trajectory["ts"].values[0]

    total_time = time.strftime("%H:%M:%S", time.gmtime(total_time))

    # add title to figure
    fig.suptitle(
        f"Leg: {domain.origin} -> {domain.destination} \n A/C perf. model: {domain.aircraft_state.performance_model_type.name}; Fuel: {total_fuel} Kg; Time: {total_time}",
        fontsize=16,
    )

    plt.tight_layout()

    return fig


def plot_trajectory(lat1, lon1, lat2, lon2, trajectory: pd.DataFrame) -> Figure:
    """Plot the trajectory of an object

    # Parameters
        trajectory (pd.DataFrame): the trajectory of the object

    # Returns
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

    # Parameters
        trajectory (pd.DataFrame): the trajectory of the object

    # Returns
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


def plot_mass(trajectory: pd.DataFrame) -> Figure:
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
    dist = np.array([d.distanceTo(pos[0]) for d in pos])

    ax.plot(dist / nm, trajectory.mass)
    ax.set_xlabel("ESAD (nm)")
    ax.set_ylabel("Mass (Kg)")

    return fig


def plot_altitude(trajectory: pd.DataFrame) -> Figure:
    fig = plt.Figure()
    ax = plt.axes()
    pos = [
        LatLon(
            trajectory.iloc[i]["lat"],
            trajectory.iloc[i]["lon"],
            trajectory.iloc[i]["alt"] * ft,
        )
        for i in range(len(trajectory.alt))
    ]
    dist = np.array([d.distanceTo(pos[0]) for d in pos])

    ax.plot(dist / nm, trajectory.alt)
    ax.set_xlabel("ESAD (nm)")
    ax.set_ylabel("Zp (ft)")

    return fig


def plot_network(domain, dir=None):
    network = domain.network

    fig = plt.figure(figsize=(10, 7))

    # define the grid layout
    gs = gridspec.GridSpec(1, 2)

    # add subplots for the line plots
    ax1 = fig.add_subplot(gs[0])

    # plot the altitude
    for node in network.nodes:
        ax1.scatter(
            network.nodes[node]["ts"],
            network.nodes[node]["height"] / ft,
            color="k",
            s=0.5,
        )

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Zp (ft)")
    ax1.set_title("Altitude profile")

    # plot the trajectory
    latmin, latmax = min(domain.lat1, domain.lat2), max(domain.lat1, domain.lat2)
    lonmin, lonmax = min(domain.lon1, domain.lon2), max(domain.lon1, domain.lon2)

    ax3 = fig.add_subplot(gs[1], projection=ccrs.PlateCarree())

    ax3.set_extent([lonmin - 3, lonmax + 3, latmin - 2, latmax + 2])

    ax3.add_feature(BORDERS, lw=0.5, color="gray")
    ax3.gridlines(draw_labels=True, color="gray", alpha=0.5, ls="--")
    ax3.coastlines(resolution="50m", lw=0.5, color="gray")

    for node in network.nodes:
        ax3.scatter(
            network.nodes[node]["lon"],
            network.nodes[node]["lat"],
            transform=ccrs.Geodetic(),
            color="blue",
            s=1,
        )

    # plot airports
    lat_start, lon_start = domain.lat1, domain.lon1
    lat_end, lon_end = domain.lat2, domain.lon2

    ax3.scatter(lon_start, lat_start, transform=ccrs.Geodetic(), color="red", s=3)
    ax3.scatter(lon_end, lat_end, transform=ccrs.Geodetic(), color="red", s=3)

    # # plot the edges
    # for edge in network.edges:
    #     # print(f"{edge[0]} -> {edge[1]}")
    #     ax3.plot(
    #         [network.nodes[edge[0]]["lon"], network.nodes[edge[1]]["lon"]],
    #         [network.nodes[edge[0]]["lat"], network.nodes[edge[1]]["lat"]],
    #         transform=ccrs.Geodetic(),
    #         color='black',
    #         lw=0.5
    #     )

    plt.tight_layout()
    plt.show()

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
