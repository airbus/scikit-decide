from pygeodesy.ellipsoidalVincenty import LatLon
from openap.extra.aero import bearing, distance
import networkx as nx
from typing import List, Tuple, Dict
import math
from openap.extra.aero import ft

# --- Simplified calculate_grid_point_coords (reverting to a version closer to initial intent) ---
def calculate_grid_point_coords(
    p0: LatLon,
    p1: LatLon,
    x_local_km: float,
    y_local_km: float,
    z_local_ft: float
) -> Tuple[float, float, float]:
    """
    Calculates the geographical coordinates of a grid point.
    Lateral displacement is perpendicular to the *current* tangent of the great circle.
    This method will show "fanning out" over long distances but is straightforward
    for localized grid generation and matches your visual examples.

    Args:
        p0 (LatLon): The actual start point of the main flight.
        p1 (LatLon): The actual end point of the main flight.
        x_local_km (float): Distance along the main path from p0.
        y_local_km (float): Lateral offset in km (negative for left, positive for right).
        z_local_ft (float): Altitude in feet for this grid point.

    Returns:
        Tuple[float, float, float]: A tuple containing (latitude, longitude, altitude in feet)
                                    for the calculated grid point.
    """
    # Calculate initial bearing from p0 to p1 to determine the direction of the main path
    initial_path_bearing = bearing(p0.lat, p0.lon, p1.lat, p1.lon)

    # Find the point along the main great circle path based on x_local_km
    point_on_path = p0.destination(distance=x_local_km * 1000, bearing=initial_path_bearing)

    # Determine the current bearing of the geodesic at 'point_on_path' towards p1
    # This is crucial for keeping lateral offsets "perpendicular" to the path's current direction.
    current_path_bearing = bearing(point_on_path.lat, point_on_path.lon, p1.lat, p1.lon)

    # Apply lateral offset from 'point_on_path'
    if y_local_km != 0:
        lateral_bearing = (current_path_bearing + (90 if y_local_km > 0 else -90)) % 360
        grid_point = point_on_path.destination(distance=abs(y_local_km) * 1000,
                                               bearing=lateral_bearing)
    else:
        grid_point = point_on_path # No lateral offset, point is directly on the main path

    return grid_point.lat, grid_point.lon, z_local_ft


def create_flight_graph(
    p0: LatLon,
    p1: LatLon,
    nb_forward_points: int = 10,
    nb_lateral_points: int = 5,
    nb_climb_descent_steps: int = 3,
    flight_levels_ft: List[float] = [32000.0],
    graph_width: str = "medium"
) -> nx.DiGraph:
    """
    Creates a symmetric 3D directed graph representing potential flight paths between two points,
    with phase-aware node generation and edge connectivity.
    """

    start_alt_ft = p0.height / 0.3048
    end_alt_ft = p1.height / 0.3048
    first_cruise_ft = flight_levels_ft[0] if flight_levels_ft else 35000.0

    need_climb = start_alt_ft < first_cruise_ft
    need_descent = end_alt_ft < first_cruise_ft

    nb_climb_steps = nb_climb_descent_steps if need_climb else 0
    nb_descent_steps = nb_climb_descent_steps if need_descent else 0

    # --- Setup ---
    if graph_width == "medium": y_grid_width_km = 1000.0
    elif graph_width == "wide": y_grid_width_km = 2000.0
    else: y_grid_width_km = 500.0
    total_x_distance_km = p0.distanceTo(p1) / 1000.0
    if not flight_levels_ft: raise ValueError("flight_levels_ft cannot be empty.")
    flight_levels_ft.sort()
    x_step_km = total_x_distance_km / (nb_forward_points - 1) if nb_forward_points > 1 else total_x_distance_km
    y_step_km = y_grid_width_km / (nb_lateral_points - 1) if nb_lateral_points > 1 else y_grid_width_km
    G = nx.DiGraph()
    y_center_idx = nb_lateral_points // 2
    min_forward_points_needed = 1 + nb_climb_steps + 1 + nb_descent_steps + 1
    if nb_forward_points < min_forward_points_needed:
        nb_forward_points = min_forward_points_needed
    climb_phase_end_x_idx = nb_climb_steps
    descent_phase_start_x_idx = nb_forward_points - 1 - nb_descent_steps

    # --- Altitude Definitions ---
    climb_altitudes = set()
    cruise_altitudes = set(flight_levels_ft)
    descent_altitudes = set()
    if nb_climb_steps > 0 and start_alt_ft < min(flight_levels_ft):
        climb_alt_step_val = (min(flight_levels_ft) - start_alt_ft) / nb_climb_steps
        for step in range(1, nb_climb_steps + 1):
            climb_altitudes.add(round(start_alt_ft + step * climb_alt_step_val, 0))
    if nb_descent_steps > 0 and end_alt_ft < min(flight_levels_ft):
        descent_alt_step_val = (min(flight_levels_ft) - end_alt_ft) / nb_descent_steps
        for step in range(1, nb_descent_steps + 1):
            descent_altitudes.add(round(min(flight_levels_ft) - (step * descent_alt_step_val), 0))
    start_end_altitudes = {round(start_alt_ft, 0), round(end_alt_ft, 0)}
    all_possible_altitudes = climb_altitudes | cruise_altitudes | descent_altitudes | start_end_altitudes
    sorted_altitudes = sorted(all_possible_altitudes)
    altitude_to_z_idx = {alt: i for i, alt in enumerate(sorted_altitudes)}
    z_idx_to_altitude = {i: alt for i, alt in enumerate(sorted_altitudes)}
    max_z_idx = len(sorted_altitudes) - 1

    # --- Node Generation ---
    start_z_idx = altitude_to_z_idx[round(start_alt_ft, 0)]
    start_node_key = (0, y_center_idx, start_z_idx)
    G.add_node(start_node_key, latlon=p0, flight_level=round(start_alt_ft, 0) * ft, phase="start")

    if need_climb:
        climb_altitudes_sorted = sorted(climb_altitudes)
        for i in range(1, climb_phase_end_x_idx + 1):
            x_local_km = i * x_step_km
            climb_step_ratio = i / nb_climb_steps
            step_idx = min(int(climb_step_ratio * len(climb_altitudes_sorted)), len(climb_altitudes_sorted) - 1)
            target_alt_ft = climb_altitudes_sorted[step_idx]
            current_z_idx_for_climb = altitude_to_z_idx[target_alt_ft]
            num_active_lateral_points = 1 + int((nb_lateral_points - 1) * (i / climb_phase_end_x_idx))
            active_y_indices = []
            if num_active_lateral_points == 1: active_y_indices.append(y_center_idx)
            else:
                start_y_idx_for_active = y_center_idx - (num_active_lateral_points // 2)
                for k in range(num_active_lateral_points): active_y_indices.append(start_y_idx_for_active + k)
            for j in active_y_indices:
                y_local_km = -y_grid_width_km / 2 + j * y_step_km
                node_key = (i, j, current_z_idx_for_climb)
                lat, lon, _ = calculate_grid_point_coords(p0, p1, x_local_km, y_local_km, target_alt_ft)
                G.add_node(node_key, latlon=LatLon(lat, lon, height=target_alt_ft * 0.3048), flight_level=target_alt_ft, phase="climb")

    cruise_altitudes_sorted = sorted(cruise_altitudes)
    for i in range(climb_phase_end_x_idx, descent_phase_start_x_idx + 1):
        if not need_descent and i == (nb_forward_points - 1):
            continue

        x_local_km = i * x_step_km
        for j in range(nb_lateral_points):
            y_local_km = -y_grid_width_km / 2 + j * y_step_km
            for fl_ft in cruise_altitudes_sorted:
                if (i == climb_phase_end_x_idx and need_climb and fl_ft != first_cruise_ft) or \
                   (i == descent_phase_start_x_idx and need_descent and fl_ft != first_cruise_ft):
                    continue
                node_key = (i, j, altitude_to_z_idx[fl_ft])
                lat, lon, _ = calculate_grid_point_coords(p0, p1, x_local_km, y_local_km, fl_ft)
                G.add_node(node_key, latlon=LatLon(lat, lon, height=fl_ft * 0.3048), flight_level=fl_ft, phase="cruise")

    if need_descent:
        descent_alt_step_val = (min(flight_levels_ft) - end_alt_ft) / max(nb_descent_steps, 1)
        for i in range(descent_phase_start_x_idx + 1, nb_forward_points - 1):
            x_local_km = i * x_step_km
            descent_step_progress = i - descent_phase_start_x_idx
            target_alt_ft_raw = min(flight_levels_ft) - (descent_step_progress * descent_alt_step_val)
            target_alt_ft = round(max(end_alt_ft, target_alt_ft_raw), 0)
            if target_alt_ft not in altitude_to_z_idx: continue
            current_z_idx_for_descent = altitude_to_z_idx[target_alt_ft]
            steps_remaining_to_end = (nb_forward_points - 1) - i
            lateral_compression_factor = (steps_remaining_to_end / nb_descent_steps) if nb_descent_steps > 0 else 0
            num_active_lateral_points = 1 + int((nb_lateral_points - 1) * lateral_compression_factor)
            active_y_indices = []
            if num_active_lateral_points == 1: active_y_indices.append(y_center_idx)
            else:
                start_y_idx_for_active = y_center_idx - (num_active_lateral_points // 2)
                for k in range(num_active_lateral_points): active_y_indices.append(start_y_idx_for_active + k)
            for j in active_y_indices:
                y_local_km = -y_grid_width_km / 2 + j * y_step_km
                node_key = (i, j, current_z_idx_for_descent)
                lat, lon, _ = calculate_grid_point_coords(p0, p1, x_local_km, y_local_km, target_alt_ft)
                G.add_node(node_key, latlon=LatLon(lat, lon, height=target_alt_ft * 0.3048), flight_level=target_alt_ft, phase="descent")

    end_z_idx = altitude_to_z_idx[round(end_alt_ft, 0)]
    end_node_key = (nb_forward_points - 1, y_center_idx, end_z_idx)
    G.add_node(end_node_key, latlon=p1, flight_level=round(end_alt_ft, 0)*ft, phase="end")

    # --- Add Edges (No Changes from your last version) ---
    for target_y_idx in range(nb_lateral_points):
        for target_z_idx_option in range(max_z_idx + 1):
            target_node_key = (1, target_y_idx, target_z_idx_option)
            if target_node_key in G:
                G.add_edge(start_node_key, target_node_key)

    for i in range(1, nb_forward_points - 1):
        for j in range(nb_lateral_points):
            for k in range(max_z_idx + 1):
                source_node_key = (i, j, k)
                if source_node_key not in G: continue
                current_alt_ft = G.nodes[source_node_key]['flight_level']
                allowed_dz_relative = set()

                if need_climb and i < climb_phase_end_x_idx:
                    i_next = i + 1
                    climb_altitudes_sorted = sorted(climb_altitudes)
                    climb_step_ratio_next = i_next / nb_climb_steps
                    step_idx_next = min(int(climb_step_ratio_next * len(climb_altitudes_sorted)), len(climb_altitudes_sorted) - 1)
                    if 0 <= step_idx_next < len(climb_altitudes_sorted):
                        target_alt_for_next_step = climb_altitudes_sorted[step_idx_next]
                        dz = altitude_to_z_idx[target_alt_for_next_step] - k
                        if dz >= 0: allowed_dz_relative.add(dz)
                    if current_alt_ft < first_cruise_ft: allowed_dz_relative.add(0)

                if climb_phase_end_x_idx <= i <= descent_phase_start_x_idx:
                    if current_alt_ft in flight_levels_ft:
                        allowed_dz_relative.add(0)
                        if k + 1 <= max_z_idx and z_idx_to_altitude[k + 1] in flight_levels_ft: allowed_dz_relative.add(1)
                        if k - 1 >= 0 and z_idx_to_altitude[k - 1] in flight_levels_ft: allowed_dz_relative.add(-1)

                if i == descent_phase_start_x_idx and need_descent:
                    descent_alt_step_val = (min(flight_levels_ft) - end_alt_ft) / max(nb_descent_steps, 1)
                    descent_alt = round(min(flight_levels_ft) - descent_alt_step_val, 0)
                    if descent_alt in descent_altitudes:
                        dz = altitude_to_z_idx[descent_alt] - k
                        allowed_dz_relative.add(dz)

                if i > descent_phase_start_x_idx and need_descent:
                    descent_alt_step_val = (min(flight_levels_ft) - end_alt_ft) / max(nb_descent_steps, 1)
                    descent_step_progress_next = (i + 1) - descent_phase_start_x_idx
                    target_alt = round(min(flight_levels_ft) - descent_step_progress_next * descent_alt_step_val, 0)
                    if target_alt in descent_altitudes:
                        dz = altitude_to_z_idx[target_alt] - k
                        if dz <= 0: allowed_dz_relative.add(dz)
                    if current_alt_ft > end_alt_ft: allowed_dz_relative.add(0)

                for dy in [-1, 0, 1]:
                    target_y_idx = j + dy
                    if 0 <= target_y_idx < nb_lateral_points:
                        for dz_val in allowed_dz_relative:
                            target_z_idx = k + dz_val
                            if 0 <= target_z_idx <= max_z_idx:
                                target_node_key = (i + 1, target_y_idx, target_z_idx)
                                if target_node_key in G:
                                    G.add_edge(source_node_key, target_node_key)

    last_layer_x_idx = nb_forward_points - 2
    for source_node_key in G.copy().nodes():
        if source_node_key[0] == last_layer_x_idx:
            G.add_edge(source_node_key, end_node_key)

    return G



def prune_graph(G: nx.DiGraph) -> nx.DiGraph:
    """
    Removes iteratively nodes that do not have any parent or child node (dead-ends),
    excluding the start and end nodes.

    Args:
        G (nx.DiGraph): The input flight graph.

    Returns:
        nx.DiGraph: The pruned graph.
    """
    G_pruned = G.copy()

    # Identify start and end node keys
    x_indices = [key[0] for key in G_pruned.nodes]
    if not x_indices:
        return G_pruned  # Empty graph

    start_node_key = min(G_pruned.nodes, key=lambda x: x[0])
    end_node_key = max(G_pruned.nodes, key=lambda x: x[0])

    removed_any = True
    while removed_any:
        removed_any = False
        to_remove = []

        for node in G_pruned.nodes:
            if node in (start_node_key, end_node_key):
                continue
            if G_pruned.in_degree(node) == 0 or G_pruned.out_degree(node) == 0:
                to_remove.append(node)

        if to_remove:
            G_pruned.remove_nodes_from(to_remove)
            removed_any = True

    return G_pruned
