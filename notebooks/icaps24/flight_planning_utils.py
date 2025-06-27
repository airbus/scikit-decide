from math import cos, pi, sin, sqrt

import folium


def plot_map(path, G, domain):
    m = folium.Map(
        location=[0.5 * (domain.lat1 + domain.lat2), 0.5 * (domain.lon1 + domain.lon2)],
        zoom_start=5,
    )

    for f in G.nodes:
        folium.Marker(
            location=[
                domain.network.nodes[f]["lat"],
                domain.network.nodes[f]["lon"],
            ],
            popup=str(f),
            icon=folium.Icon(color="beige"),
        ).add_to(m)

    for (f, t) in G.edges:
        folium.PolyLine(
            locations=[
                (
                    domain.network.nodes[f]["lat"],
                    domain.network.nodes[f]["lon"],
                ),
                (
                    domain.network.nodes[t]["lat"],
                    domain.network.nodes[t]["lon"],
                ),
            ],
            color="beige",
        ).add_to(m)

    folium.Marker(
        location=[domain.lat1, domain.lon1],
        popup="origin",
        icon=folium.Icon(color="blue"),
    ).add_to(m)
    folium.Marker(
        location=[domain.lat2, domain.lon2],
        popup="arrival",
        icon=folium.Icon(color="red"),
    ).add_to(m)

    for i in range(len(path) - 1):
        p = path[i]
        pp = path[i + 1]
        folium.Marker(
            location=[
                domain.network.nodes[p]["lat"],
                domain.network.nodes[p]["lon"],
            ],
            popup=str(p),
            icon=folium.Icon(color="green"),
        ).add_to(m)
        folium.PolyLine(
            locations=[
                (
                    domain.network.nodes[p]["lat"],
                    domain.network.nodes[p]["lon"],
                ),
                (
                    domain.network.nodes[pp]["lat"],
                    domain.network.nodes[pp]["lon"],
                ),
            ],
            color="green",
        ).add_to(m)
    return m


def cost(domain, f, t):
    EARTH_RADIUS = 3440
    wp1 = domain.network.nodes[f]
    wp2 = domain.network.nodes[t]
    lat1 = wp1["lat"]
    lon1 = wp1["lon"]
    height1 = wp1["height"]
    lat2 = wp2["lat"]
    lon2 = wp2["lon"]

    AIRCRAFT_SPEED = 500
    WIND_DIRECTION = (
        domain.weather_interpolator.interpol_wind_classic(lat1, lon1, 0, 34 * height1)[
            1
        ]
        * 180
    )
    WIND_SPEED = domain.weather_interpolator.interpol_wind_classic(
        lat1, lon1, 0, 34 * height1
    )[0]
    # Computes coordinates of the direction vector in the Earth-centered system
    dir_x = EARTH_RADIUS * (
        (cos(lat2 * pi / 180.0) * cos(lon2 * pi / 180.0))
        - (cos(lat1 * pi / 180.0) * cos(lon1 * pi / 180.0))
    )
    dir_y = EARTH_RADIUS * (
        (cos(lat2 * pi / 180.0) * sin(lon2 * pi / 180.0))
        - (cos(lat1 * pi / 180.0) * sin(lon1 * pi / 180.0))
    )
    dir_z = EARTH_RADIUS * (sin(lat2 * pi / 180.0) - sin(lat1 * pi / 180.0))
    # Computes coordinates of the direction vector in the tangential plane at the waypoint node.data
    dir_a = (-dir_x * sin(lon1 * pi / 180.0)) + (dir_y * cos(lon1 * pi / 180.0))
    dir_b = (
        (dir_x * (-sin(lat1 * pi / 180.0) * cos(lon1 * pi / 180.0)))
        + (dir_y * (-sin(lat1 * pi / 180.0) * sin(lon1 * pi / 180.0)))
        + (dir_z * cos(lat1 * pi / 180.0))
    )
    # Normalize the direction vector
    dir_na = dir_a / sqrt(dir_a * dir_a + dir_b * dir_b)
    dir_nb = dir_b / sqrt(dir_a * dir_a + dir_b * dir_b)
    # Compute wind vector in the tangential plane
    w_a = WIND_SPEED * sin(WIND_DIRECTION * pi / 180.0)
    w_b = WIND_SPEED * cos(WIND_DIRECTION * pi / 180.0)
    # Compute speed along direction vector
    mu = (dir_na * w_a) + (dir_nb * w_b)
    phi = (mu * mu) - (WIND_SPEED * WIND_SPEED) + (AIRCRAFT_SPEED * AIRCRAFT_SPEED)
    dir_speed = mu + sqrt(phi)
    flown_distance = AIRCRAFT_SPEED * sqrt(dir_a * dir_a + dir_b * dir_b) / dir_speed
    return flown_distance
