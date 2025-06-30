# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import time
import networkx as nx
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
import folium

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

def plot_graph_vertical(
        G: nx.DiGraph, 
        title: str = "Flight Graph Vertical Profile"):
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)

    # Plot all edges (as lines between nodes in vertical profile)
    for u, v in G.edges():
        x1, _, z1 = u
        x2, _, z2 = v
        alt1 = G.nodes[u]['flight_level']
        alt2 = G.nodes[v]['flight_level']
        ax.plot([x1, x2], [alt1, alt2], c='gray', alpha=0.3, linewidth=1)

    # Plot all nodes
    x_indices = []
    altitudes = []
    for node_key, data in G.nodes(data=True):
        x_indices.append(node_key[0])
        altitudes.append(data["flight_level"])
    ax.scatter(x_indices, altitudes, c='blue', s=20, alpha=0.7, label='Nodes')

    # Highlight Start Node
    y_center_idx = max(n[1] for n in G.nodes) // 2
    start_node_key = (0, y_center_idx, 0)
    if start_node_key in G.nodes:
        start_data = G.nodes[start_node_key]
        ax.scatter(start_node_key[0], start_data['flight_level'],
                   c='green', s=120, marker='^', label='Start Node')

    # Highlight End Node
    max_x_index = max(n[0] for n in G.nodes)
    end_node_key = (max_x_index, y_center_idx, 0)
    if end_node_key in G.nodes:
        end_data = G.nodes[end_node_key]
        ax.scatter(end_node_key[0], end_data['flight_level'],
                   c='red', s=120, marker='X', label='End Node')

    ax.set_xlabel('Forward Point Index (X-axis in graph)')
    ax.set_ylabel('Altitude (ft)')
    ax.set_title(title)
    # ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def plot_flight_graph_on_map_with_edges(
        G: nx.DiGraph, 
        nb_lateral_points_param: int) -> folium.Map:
    # Extract lat/lon
    lats = [data['latlon'].lat for _, data in G.nodes(data=True)]
    lons = [data['latlon'].lon for _, data in G.nodes(data=True)]
    center_lat = sum(lats) / len(lats) if lats else 0.0
    center_lon = sum(lons) / len(lons) if lons else 0.0

    # Initialize map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles='CartoDB positron')

    # ✅ Draw edges correctly between connected nodes
    for u, v in G.edges():
        u_data = G.nodes[u]
        v_data = G.nodes[v]

        try:
            u_latlon = u_data['latlon']
            v_latlon = v_data['latlon']
            folium.PolyLine(
                locations=[(u_latlon.lat, u_latlon.lon), (v_latlon.lat, v_latlon.lon)],
                color='gray',
                weight=1,
                opacity=0.4
            ).add_to(m)
        except KeyError:
            continue  # Skip if latlon is missing

    # 🔵 Plot all nodes
    for node, data in G.nodes(data=True):
        folium.CircleMarker(
            location=(data['latlon'].lat, data['latlon'].lon),
            radius=2,
            color='blue',
            fill=True,
            fill_opacity=0.6,
            tooltip=f"FL: {data['flight_level']:.0f}ft (Node: {node})"
        ).add_to(m)

    # ✅ Start node marker
    y_center_idx = nb_lateral_points_param // 2
    start_node_key = (0, y_center_idx, 0)
    if start_node_key in G.nodes:
        start = G.nodes[start_node_key]
        folium.Marker(
            location=(start['latlon'].lat, start['latlon'].lon),
            popup='Start',
            icon=folium.Icon(color='green', icon='play')
        ).add_to(m)

    # ✅ End node marker
    max_x = max(n[0] for n in G.nodes) if G.nodes else 0
    end_node_key = (max_x, y_center_idx, 0)
    if end_node_key in G.nodes:
        end = G.nodes[end_node_key]
        folium.Marker(
            location=(end['latlon'].lat, end['latlon'].lon),
            popup='End',
            icon=folium.Icon(color='red', icon='stop')
        ).add_to(m)

    return m

def plot_full(domain, trajectory: pd.DataFrame) -> Figure:
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
        f"Leg: {str(domain.origin)} -> {str(domain.destination)} \n"
        f"A/C perf. model: {domain.initial_aircraft_state.performance_model_type.name}; "
        f"Fuel: {total_fuel} Kg; Time: {total_time}",
        fontsize=16,
    )

    plt.tight_layout()

    return fig

def plot_network_adapted(graph: nx.DiGraph, p0: LatLon, p1: LatLon, dir_path: str = None):
    """
    Plots the vertical profile and map trajectory of a given flight graph.
    This version is adapted to work with the graph from create_flight_graph.

    Args:
        graph (nx.DiGraph): The flight graph to plot.
        p0 (LatLon): The starting point object.
        p1 (LatLon): The ending point object.
        dir_path (str, optional): Directory to save the plot image. Defaults to None.
    """
    if not graph or graph.number_of_nodes() == 0:
        print("Graph is empty, cannot plot.")
        return

    fig = plt.figure(figsize=(20, 9))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.5])  # Give more space to the map

    # --- 1. Altitude Profile Plot ---
    ax1 = fig.add_subplot(gs[0])

    # Plot edges first to form the background structure
    for u, v in graph.edges():
        x_edge = [u[0], v[0]]
        y_edge = [graph.nodes[u]['flight_level'], graph.nodes[v]['flight_level']]
        ax1.plot(x_edge, y_edge, color="gray", alpha=0.5, linewidth=0.7)

    # Plot nodes on top of the edges
    # x-axis is the forward point index from the node key: (x, y, z) -> x
    # y-axis is the flight level in feet from the node attribute
    x_coords = [node[0] for node in graph.nodes]
    y_coords_ft = [data['flight_level'] for node, data in graph.nodes(data=True)]
    ax1.scatter(x_coords, y_coords_ft, color="blue", s=8, zorder=5, label="Nodes")

    ax1.set_xlabel("Forward Point Index")
    ax1.set_ylabel("Altitude (ft)")
    ax1.set_title("Altitude Profile")
    ax1.grid(True)
    ax1.legend()

    # --- 2. Map Trajectory Plot ---
    ax2 = fig.add_subplot(gs[1], projection=ccrs.PlateCarree())

    # Set map extent based on start and end points with some padding
    latmin, latmax = min(p0.lat, p1.lat), max(p0.lat, p1.lat)
    lonmin, lonmax = min(p0.lon, p1.lon), max(p0.lon, p1.lon)
    padding = 3  # Degrees of padding
    ax2.set_extent([lonmin - padding, lonmax + padding, latmin - padding, latmax + padding], crs=ccrs.PlateCarree())

    # Add map features
    ax2.add_feature(BORDERS, lw=0.5, color="gray")
    ax2.gridlines(draw_labels=True, color="gray", alpha=0.5, ls="--")
    ax2.coastlines(resolution="50m", lw=0.5, color="gray")

    # Plot graph edges
    for u, v in graph.edges():
        u_data = graph.nodes[u]
        v_data = graph.nodes[v]
        # Check that both nodes have the latlon attribute before plotting
        if u_data.get('latlon') and v_data.get('latlon'):
            lon_edge = [u_data['latlon'].lon, v_data['latlon'].lon]
            lat_edge = [u_data['latlon'].lat, v_data['latlon'].lat]
            ax2.plot(lon_edge, lat_edge, transform=ccrs.Geodetic(), color="gray", alpha=0.6, linewidth=0.5)

    # Plot graph nodes
    # Access lat/lon correctly from the 'latlon' object
    for node, data in graph.nodes(data=True):
        if data.get('latlon'):
            ax2.scatter(
                data["latlon"].lon,
                data["latlon"].lat,
                transform=ccrs.Geodetic(),
                color="blue",
                s=2,
                zorder=5
            )

    # Plot start and end points
    ax2.scatter(p0.lon, p0.lat, transform=ccrs.Geodetic(), color="red", s=30, zorder=10, label="Start/End")
    ax2.scatter(p1.lon, p1.lat, transform=ccrs.Geodetic(), color="red", s=30, zorder=10)
    ax2.set_title("Geographical Trajectory")
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # Save the figure
    if dir_path:
        fig.savefig(f"{dir_path}/network_plot.png", dpi=300)
    else:
        fig.savefig("network_plot.png", dpi=300)
