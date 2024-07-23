import math
from argparse import Action
from enum import Enum

# data and math
from math import asin, atan2, cos, radians, sin, sqrt

# typing
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

# plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# aircraft performance model
from openap.extra.aero import bearing as aero_bearing
from openap.extra.aero import distance, ft, kts, latlon, mach2tas
from openap.extra.nav import airport
from openap.prop import aircraft
from pygeodesy.ellipsoidalVincenty import LatLon

from skdecide import DeterministicPlanningDomain, ImplicitSpace, Solver, Space, Value
from skdecide.builders.domain import Renderable, UnrestrictedActions

# custom aircraft performance model
from skdecide.hub.domain.flight_planning.aircraft_performance.base import (
    AircraftPerformanceModel,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.poll_schumann_utils.engine_loader import (
    load_aircraft_engine_params,
)
from skdecide.hub.domain.flight_planning.flightplanning_utils import (
    plot_full,
    plot_trajectory,
)
from skdecide.hub.domain.flight_planning.weather_interpolator.weather_tools.get_weather_noaa import (
    get_weather_matrix,
)
from skdecide.hub.domain.flight_planning.weather_interpolator.weather_tools.interpolator.GenericInterpolator import (
    GenericWindInterpolator,
)
from skdecide.hub.space.gym import EnumSpace, ListSpace, MultiDiscreteSpace
from skdecide.utils import load_registered_solver

try:
    from IPython.display import clear_output as ipython_clear_output
except ImportError:
    ipython_available = False
else:
    ipython_available = True


def clear_output(wait=True):
    if ipython_available:
        ipython_clear_output(wait=wait)


class WeatherDate:
    day: int
    month: int
    year: int
    forecast: str
    leapyear: bool

    def __init__(self, day, month, year, forecast="nowcast") -> None:
        self.day = int(day)
        self.month = int(month)
        self.year = int(year)
        self.forecast = forecast
        self.leapyear = self.year % 400 == 0 or (
            self.year % 100 != 0 and self.year % 4 == 0
        )

    def __hash__(self) -> int:
        return hash((self.day, self.month, self.year, self.forecast))

    def __eq__(self, other: object) -> bool:
        return (
            self.day == other.day
            and self.month == other.month
            and self.year == other.year
            and self.forecast == other.forecast
        )

    def __ne__(self, other: object) -> bool:
        return (
            self.day != other.day
            or self.month != other.month
            or self.year != other.year
            or self.forecast != other.forecast
        )

    def __str__(self) -> str:
        day = str(self.day)
        month = str(self.month)

        if len(day) == 1:
            day = "0" + day
        if len(month) == 1:
            month = "0" + month

        return f"[{day} {month} {self.year}, forecast : {self.forecast}]"

    def to_dict(self) -> dict:
        day = str(self.day)
        month = str(self.month)

        if len(day) == 1:
            day = "0" + day
        if len(month) == 1:
            month = "0" + month

        return {
            "year": str(self.year),
            "month": str(month),
            "day": str(day),
            "forecast": self.forecast,
        }

    def next_day(self):
        day = self.day
        month = self.month
        year = self.year
        if month == 12 and day == 31:
            year += 1
            month = 1
            day = 1

        elif month in (1, 3, 5, 7, 8, 10) and day == 31:
            day = 1
            month += 1

        elif month in (4, 6, 9, 11) and day == 30:
            day = 1
            month += 1

        elif month == 2:
            if (self.leap_year and day == 29) or (not (self.leap_year) and day == 28):
                day = 1
                month = 3
            else:
                day += 1

        else:
            day += 1

        return WeatherDate(day, month, year, forecast=self.forecast)

    def previous_day(self):
        day = self.day
        month = self.month
        year = self.year
        if month == 1 and day == 1:
            year -= 1
            month = 12
            day = 31

        elif month in (5, 7, 10, 12) and day == 1:
            day = 30
            month -= 1

        elif month in (2, 4, 6, 8, 9, 11) and day == 1:
            day = 31
            month -= 1

        elif month == 3 and day == 1:
            if self.leap_year:
                day = 29
                month = 2
            else:
                day = 28
                month = 2

        else:
            day -= 1

        return WeatherDate(day, month, year, forecast=self.forecast)


class State:
    """
    Definition of a aircraft state during the flight plan
    """

    trajectory: pd.DataFrame
    pos: Tuple[int, int, int]

    def __init__(self, trajectory, pos):
        """Initialisation of a state

        # Parameters
            trajectory : Trajectory information of the flight
            pos : Current position in the airways graph
        """
        self.trajectory = trajectory
        self.pos = pos
        if trajectory is not None:
            self.mass = trajectory.iloc[-1]["mass"]
            self.alt = trajectory.iloc[-1]["alt"]
            self.time = trajectory.iloc[-1]["ts"]
        else:
            self.mass = None
            self.alt = None
            self.time = None

    def __hash__(self):
        # print(self.pos, self.mass, self.alt, self.time)
        return hash((self.pos, int(self.mass), self.alt, int(self.time)))

    def __eq__(self, other):
        return (
            self.pos == other.pos
            and int(self.mass) == int(other.mass)
            and self.alt == other.alt
            and int(self.time) == int(other.time)
        )

    def __ne__(self, other):
        return (
            self.pos != other.pos
            or int(self.mass) != int(other.mass)
            or self.alt != other.alt
            or int(self.time) != int(other.time)
        )

    def __str__(self):
        return f"[{self.trajectory.iloc[-1]['ts']:.2f} \
            {self.pos} \
            {self.trajectory.iloc[-1]['alt']:.2f} \
            {self.trajectory.iloc[-1]['mass']:.2f}]"


class H_Action(Enum):
    """
    Horizontal action that can be perform by the aircraft
    """

    up = -1
    straight = 0
    down = 1


class V_Action(Enum):
    """
    Vertical action that can be perform by the aircraft
    """

    climb = 1
    cruise = 0
    descent = -1


class D(DeterministicPlanningDomain, UnrestrictedActions, Renderable):
    T_state = State  # Type of states
    T_observation = State  # Type of observations
    T_event = Action  # Type of events
    T_value = float  # Type of transition values (rewards or costs)
    T_predicate = bool  # Type of transition predicates (terminal states)
    T_info = None  # Type of additional information in environment outcome
    T_agent = Union  # Type of agent


class FlightPlanningDomain(
    DeterministicPlanningDomain, UnrestrictedActions, Renderable
):
    """Automated flight planning domain.

    Domain definition
    -----------------

    The flight planning domain can be quickly defined as :

    - An origin, as ICAO code of an airport,
    - A destination, as ICAO code of an airport,
    - An aircraft type, as a string recognizable by the OpenAP library.

    Airways graph
    -------------

    A three-dimensional airway graph of waypoints is created. The graph is following the great circle
    which represents the shortest pass between the origin and the destination.
    The planner computes a plan by choosing waypoints in the graph, which are represented by 4-dimensionnal states.
    There is 3 phases in the graph :

    - The climbing phase
    - The cruise phase
    - The descent phase

    The flight planning domain allows to choose a number of forward, lateral and vertical waypoints in the graph.
    It is also possible to choose different width (tiny, small, normal, large, xlarge) which will increase
    or decrease the graph width.

    State representation
    --------------------

    Here, the states are represented by 4 features :

    - The position in the graph (x,y,z)
    - The aircraft mass, which can also represent the fuel consumption (integer)
    - The altitude (integer)
    - The time (seconds)

    Wind interpolation
    ------------------

    The flight planning domain can take in consideration the wind conditions.
    That interpolation have a major impact on the results, as jet streams are high altitude wind
    which can increase or decrease the ground speed of the aircraft.
    It also have an impact on the computation time of a flight plan,
    as the objective and heuristic function became more complex.

    Objective (or cost) functions
    -----------------------------

    There is three possible objective functions:

    - Fuel (Default)
    - Distance
    - Time

    The chosen objective will represent the cost to go from a state to another. The aim of the algorithm is to minimize the cost.

    Heuristic functions
    -------------------

    When using an A* algorithm to compute the flight plan, we need to feed it with a heuristic function, which guide the algorithm.
    For now, there is 5 different (not admissible) heuristic function, depending on `self.heuristic_name`:

    - fuel, which computes the required fuel to get to the goal. It takes in consideration the local wind & speed of the aircraft.
    - time, which computes the required time to get to the goal. It takes in consideration the local wind & speed of the aircraft.
    - distance, wich computes the distance to the goal.
    - lazy_fuel, which propagates the fuel consummed so far.
    - lazy_time, which propagates the time spent on the flight so far
    - None : we give a 0 cost value, which will transform the A* algorithm into a Dijkstra-like algorithm.

    Aircraft performance models
    --------------------------

    The flight planning domain can use two possible A/C performance models:

    - OpenAP: the aircraft performance model is based on the OpenAP library.
    - Poll-Schumann: the aircraft performance model is based on Poll-Schumann equations as stated on the paper: "An estimation
    method for the fuel burn and other performance characteristics of civil transport aircraft in the cruise" by Poll and Schumann;
    The Aernautical Journal, 2020.

    Optional features
    -----------------

    The flight planning domain has several optional features :

    - Fuel loop: this is an optimisation of the loaded fuel for the aircraft.
      It will run some flights to computes the loaded fuel, using distance objective & heuristic.

    - Constraints definition: you can define constraints such as

        - A time constraint, represented by a time windows
        - A fuel constraint, represented by the maximum fuel for instance.

    - Slopes: you can define your own climbing & descending slopes which have to be between 10.0 and 25.0.

    """

    T_state = State  # Type of states
    T_observation = State  # Type of observations
    T_event = Tuple[H_Action, V_Action]  # Type of events
    T_value = float  # Type of transition values (rewards or costs)
    T_predicate = bool  # Type of transition predicates (terminal states)
    T_info = None  # Type of additional information in environment outcome
    T_agent = Union  # Type of agent

    def __init__(
        self,
        origin: Union[str, tuple],
        destination: Union[str, tuple],
        actype: str,
        weather_date: Optional[WeatherDate] = None,
        wind_interpolator: Optional[GenericWindInterpolator] = None,
        objective: str = "fuel",
        heuristic_name: str = "fuel",
        perf_model_name: str = "openap",
        constraints=None,
        nb_forward_points: int = 41,
        nb_lateral_points: int = 11,
        nb_vertical_points: Optional[int] = None,
        take_off_weight: Optional[int] = None,
        fuel_loaded: Optional[float] = None,
        fuel_loop: bool = False,
        fuel_loop_solver_cls: Optional[Type[Solver]] = None,
        fuel_loop_solver_kwargs: Optional[Dict[str, Any]] = None,
        fuel_loop_tol: float = 1e-3,
        climbing_slope: Optional[float] = None,
        descending_slope: Optional[float] = None,
        graph_width: Optional[str] = None,
        res_img_dir: Optional[str] = None,
        starting_time: float = 3_600.0 * 8.0,
    ):
        """Initialisation of a flight planning instance

        # Parameters
            origin (Union[str, tuple]):
                ICAO code of the airport, or a tuple (lat,lon,alt), of the origin of the flight plan. Altitude should be in ft
            destination (Union[str, tuple]):
                ICAO code of the airport, or a tuple (lat,lon,alt), of the destination of the flight plan. Altitude should be in ft
            actype (str):
                Aircarft type describe in openap datas (https://github.com/junzis/openap/tree/master/openap/data/aircraft)
            weather_date (WeatherDate, optional):
                Date for the weather, needed for days management.
                If None, no wind will be applied.
            wind_interpolator (GenericWindInterpolator, optional):
                Wind interpolator for the flight plan. If None, create one from the specified weather_date.
                The data is either already present locally or be downloaded from https://www.ncei.noaa.gov
            objective (str, optional):
                Cost function of the flight plan. It can be either fuel, distance or time. Defaults to "fuel".
            heuristic_name (str, optional):
                Heuristic of the flight plan, it will guide the aircraft through the graph. It can be either fuel, distance or time. Defaults to "fuel".
            perf_model_name (str, optional):
                Aircraft performance model used in the flight plan. It can be either openap or PS (Poll-Schumann). Defaults to "openap".
            constraints (_type_, optional):
                Constraints dictionnary (keyValues : ['time', 'fuel'] ) to be defined in for the flight plan. Defaults to None.
            nb_points_forward (int, optional):
                Number of forward nodes in the graph. Defaults to 41.
            nb_points_lateral (int, optional):
                Number of lateral nodes in the graph. Defaults to 11.
            nb_points_vertical (int, optional):
                Number of vertical nodes in the graph. Defaults to None.
            take_off_weight (int, optional):
                Take off weight of the aircraft. Defaults to None.
            fuel_loaded (float, optional):
                Fuel loaded in the aricraft for the flight plan. Defaults to None.
            fuel_loop (bool, optional):
                Boolean to create a fuel loop to optimize the fuel loaded for the flight. Defaults to False
            fuel_loop_solver_cls (type[Solver], optional):
                Solver class used in the fuel loop. Defaults to LazyAstar.
            fuel_loop_solver_kwargs (Dict[str, Any], optional):
                Kwargs to initialize the solver used in the fuel loop.
            climbing_slope (float, optional):
                Climbing slope of the aircraft, has to be between 10.0 and 25.0. Defaults to None.
            descending_slope (float, optional):
                Descending slope of the aircraft, has to be between 10.0 and 25.0. Defaults to None.
            graph_width (str, optional):
                Airways graph width, in ["tiny", "small", "normal", "large", "xlarge"]. Defaults to None
            res_img_dir (str, optional):
                Directory in which images will be saved. Defaults to None
            starting_time (float, optional):
                Start time of the flight, in seconds. Defaults to 8AM (3_600.0 * 8.0)
        """

        # Initialisation of the origin and the destination
        self.origin, self.destination = origin, destination
        if isinstance(origin, str):  # Origin is an airport
            ap1 = airport(origin)
            self.lat1, self.lon1, self.alt1 = ap1["lat"], ap1["lon"], ap1["alt"]
        else:  # Origin is geographic coordinates
            self.lat1, self.lon1, self.alt1 = origin

        if isinstance(destination, str):  # Destination is an airport
            ap2 = airport(destination)
            self.lat2, self.lon2, self.alt2 = ap2["lat"], ap2["lon"], ap2["alt"]
        else:  # Destination is geographic coordinates
            self.lat2, self.lon2, self.alt2 = destination

        self.start_time = starting_time
        # Retrieve the aircraft datas in openap library
        self.actype = actype
        self.ac = aircraft(actype)

        self.mach = self.ac["cruise"]["mach"]

        # Initialisation of the objective & heuristic, the constraints and the wind interpolator
        if heuristic_name in (
            "distance",
            "fuel",
            "lazy_fuel",
            "time",
            "lazy_time",
            None,
        ):
            self.heuristic_name = heuristic_name
        else:
            self.heuristic_name = "fuel"

        if objective in ("distance", "fuel", "time"):
            self.objective = objective
        else:
            self.objective = "fuel"
        self.constraints = constraints

        self.weather_date = weather_date
        self.initial_date = weather_date

        if wind_interpolator is None:
            self.weather_interpolator = self.get_weather_interpolator()

        # Build network between airports
        if graph_width:
            all_graph_width = {
                "tiny": 0.5,
                "small": 0.75,
                "normal": 1.0,
                "large": 1.5,
                "xlarge": 2.0,
            }
            graph_width = all_graph_width[graph_width]
        else:
            graph_width = 1.0

        self.nb_forward_points = nb_forward_points
        self.nb_lateral_points = nb_lateral_points

        if nb_vertical_points:
            self.nb_vertical_points = nb_vertical_points
        else:
            self.nb_vertical_points = (
                int((self.ac["limits"]["ceiling"] - self.ac["cruise"]["height"]) / 1000)
                + 1
            )
        self.network = self.set_network(
            LatLon(self.lat1, self.lon1, self.alt1 * ft),  # alt ft -> meters
            LatLon(self.lat2, self.lon2, self.alt2 * ft),  # alt ft -> meters
            self.nb_forward_points,
            self.nb_lateral_points,
            self.nb_vertical_points,
            descending_slope=descending_slope,
            climbing_slope=climbing_slope,
            graph_width=graph_width,
        )

        self.fuel_loaded = fuel_loaded

        # Initialisation of the flight plan, with the initial state
        if fuel_loop:
            if fuel_loop_solver_cls is None:
                LazyAstar = load_registered_solver("LazyAstar")
                fuel_loop_solver_cls = LazyAstar
                fuel_loop_solver_kwargs = dict(heuristic=lambda d, s: d.heuristic(s))
            elif fuel_loop_solver_kwargs is None:
                fuel_loop_solver_kwargs = {}
            fuel_loaded = fuel_optimisation(
                origin=origin,
                destination=destination,
                actype=self.actype,
                constraints=constraints,
                weather_date=weather_date,
                solver_cls=fuel_loop_solver_cls,
                solver_kwargs=fuel_loop_solver_kwargs,
                fuel_tol=fuel_loop_tol,
            )
            # Adding fuel reserve (but we can't put more fuel than maxFuel)
            fuel_loaded = min(1.1 * fuel_loaded, self.ac["limits"]["MFC"])
        elif fuel_loaded:
            self.constraints["fuel"] = (
                0.97 * fuel_loaded
            )  # Update of the maximum fuel there is to be used
        else:
            fuel_loaded = self.ac["limits"]["MFC"]

        self.fuel_loaded = fuel_loaded

        assert (
            fuel_loaded <= self.ac["limits"]["MFC"]
        )  # Ensure fuel loaded <= fuel capacity

        aircraft_params = load_aircraft_engine_params(actype)

        self.start = State(
            pd.DataFrame(
                [
                    {
                        "ts": self.start_time,
                        "lat": self.lat1,
                        "lon": self.lon1,
                        "mass": (
                            aircraft_params["amass_mtow"]
                            if take_off_weight is None
                            else take_off_weight
                            - 0.8 * (self.ac["limits"]["MFC"] - self.fuel_loaded)
                        ),  # Here we compute the weight difference between the fuel loaded and the fuel capacity
                        "mach": self.mach,
                        "fuel": 0.0,
                        "alt": self.alt1,
                    }
                ]
            ),
            (0, self.nb_lateral_points // 2, 0),
        )

        self.perf_model = AircraftPerformanceModel(actype, perf_model_name)
        self.perf_model_name = perf_model_name

        self.res_img_dir = res_img_dir
        self.cruising = self.alt1 * ft >= self.ac["cruise"]["height"] * 0.98

    # Class functions

    def _get_next_state(self, memory: D.T_state, action: D.T_event) -> D.T_state:
        """Compute the next state

        # Parameters
            memory (D.T_state): The current state
            action (D.T_event): The action to perform

        # Returns
            D.T_state: The next state
        """

        trajectory = memory.trajectory.copy()

        # Set intermediate destination point
        next_x, next_y, next_z = memory.pos

        next_x += 1

        if action[0] == H_Action.up:
            next_y += 1
        if action[0] == H_Action.down:
            next_y -= 1
        if action[1] == V_Action.climb:
            next_z += 1
        if action[1] == V_Action.descent:
            next_z -= 1

        # Aircraft stays on the network
        if (
            next_x >= self.nb_forward_points
            or next_y < 0
            or next_y >= self.nb_lateral_points
            or next_z < 0
            or next_z >= self.nb_vertical_points
        ):
            return memory

        # Concatenate the two trajectories

        to_lat = self.network[next_x][next_y][next_z].lat
        to_lon = self.network[next_x][next_y][next_z].lon
        to_alt = (
            self.network[next_x][next_y][next_z].height / ft
        )  # We compute the flight with altitude in ft, whereas the altitude in the network is in meters according to LatLon

        self.cruising = (
            to_alt * ft >= self.ac["cruise"]["height"] * 0.98
        )  # Check if the plane will be cruising in the next state
        trajectory = self.flying(trajectory.tail(1), (to_lat, to_lon, to_alt))

        state = State(
            pd.concat([memory.trajectory, trajectory], ignore_index=True),
            (next_x, next_y, next_z),
        )
        return state

    def _get_transition_value(
        self,
        memory: D.T_state,
        action: D.T_event,
        next_state: Optional[D.T_state] = None,
    ) -> Value[D.T_value]:
        """
        Get the value (reward or cost) of a transition.
        Set cost to distance travelled between points

        # Parameters
            memory (D.T_state): The current state
            action (D.T_event): The action to perform
            next_state (Optional[D.T_state], optional): The next state. Defaults to None.

        # Returns
            Value[D.T_value]: Cost to go from memory to next state
        """
        assert memory != next_state, "Next state is the same as the current state"
        if self.objective == "distance":
            cost = LatLon.distanceTo(
                LatLon(
                    memory.trajectory.iloc[-1]["lat"],
                    memory.trajectory.iloc[-1]["lon"],
                    memory.trajectory.iloc[-1]["alt"] * ft,
                ),
                LatLon(
                    next_state.trajectory.iloc[-1]["lat"],
                    next_state.trajectory.iloc[-1]["lon"],
                    next_state.trajectory.iloc[-1]["alt"] * ft,
                ),
            )
        elif self.objective == "time" or self.objective == "lazy_time":
            cost = (
                next_state.trajectory.iloc[-1]["ts"] - memory.trajectory.iloc[-1]["ts"]
            )
        else:
            cost = (
                memory.trajectory.iloc[-1]["mass"]
                - next_state.trajectory.iloc[-1]["mass"]
            )
        return Value(cost=cost)

    def _get_initial_state_(self) -> D.T_state:
        """
        Get the initial state.

        Set the start position as initial state.
        """
        return self.start

    def _get_goals_(self) -> Space[D.T_observation]:
        """
        Get the domain goals space (finite or infinite set).

        Set the end position as goal.
        """
        return ImplicitSpace(lambda x: x.pos[0] == self.nb_forward_points - 1)

    def _get_terminal_state_time_fuel(self, state: State) -> dict:
        """
        Get the domain terminal state information to compare with the constraints

        # Parameters
            state (State): terminal state to retrieve the information on fuel and time.

        # Returns
            dict: dictionnary containing both fuel and time information.
        """
        fuel = 0.0
        for trajectory in state.trajectory.iloc:
            fuel += trajectory["fuel"]

        if (
            state.trajectory.iloc[-1]["ts"] < self.start_time
        ):  # The flight arrives the next day
            time = 3_600 * 24 - self.start_time + state.trajectory.iloc[-1]["ts"]
        else:
            time = state.trajectory.iloc[-1]["ts"] - self.start_time

        return {"time": time, "fuel": fuel}

    def _is_terminal(self, state: State) -> D.T_predicate:
        """
        Indicate whether a state is terminal.

        Stop an episode only when goal reached.
        """
        return state.pos[0] == self.nb_forward_points - 1

    def _get_applicable_actions_from(self, memory: D.T_state) -> Space[D.T_event]:
        """
        Get the applicable actions from a state.
        """
        x, y, z = memory.pos

        space = []
        if x < self.nb_forward_points - 1:
            space.append((H_Action.straight, V_Action.cruise))
            if z < self.nb_vertical_points - 1 and self.cruising:
                space.append((H_Action.straight, V_Action.climb))
            if z > 0 and self.cruising:
                space.append((H_Action.straight, V_Action.descent))
            if y + 1 < self.nb_lateral_points:
                space.append((H_Action.up, V_Action.cruise))
                if z < self.nb_vertical_points - 1 and self.cruising:
                    space.append((H_Action.up, V_Action.climb))
                if z > 0 and self.cruising:
                    space.append((H_Action.up, V_Action.descent))
            if y > 0:
                space.append((H_Action.down, V_Action.cruise))
                if z < self.nb_vertical_points - 1 and self.cruising:
                    space.append((H_Action.down, V_Action.climb))
                if z > 0 and self.cruising:
                    space.append((H_Action.down, V_Action.descent))

        return ListSpace(space)

    def _get_action_space_(self) -> Space[D.T_event]:
        """
        Define action space.
        """
        return EnumSpace((H_Action, V_Action))

    def _get_observation_space_(self) -> Space[D.T_observation]:
        """
        Define observation space.
        """
        return MultiDiscreteSpace(
            [self.nb_forward_points, self.nb_lateral_points, self.nb_vertical_points]
        )

    def _render_from(self, memory: State, **kwargs: Any) -> Any:
        """
        Render visually the map.

        # Returns
            matplotlib figure
        """
        return plot_trajectory(
            self.lat1,
            self.lon1,
            self.lat2,
            self.lon2,
            memory.trajectory,
        )

    def heuristic(self, s: D.T_state, heuristic_name: str = None) -> Value[D.T_value]:
        """
        Heuristic to be used by search algorithms, depending on the objective and constraints.

        # Parameters
            s (D.T_state): Actual state
            objective (str, optional): Objective function. Defaults to None.

        # Returns
            Value[D.T_value]: Heuristic value of the state.
        """

        # current position
        pos = s.trajectory.iloc[-1]

        # parameters
        lat_to, lon_to, alt_to = self.lat2, self.lon2, self.alt2
        lat_start, lon_start, alt_start = self.lat1, self.lon1, self.alt1

        if heuristic_name is None:
            heuristic_name = self.heuristic_name

        # Compute distance in meters
        distance_to_goal = LatLon.distanceTo(
            LatLon(pos["lat"], pos["lon"], height=pos["alt"] * ft),  # alt ft -> meters
            LatLon(lat_to, lon_to, height=alt_to * ft),  # alt ft -> meters
        )
        distance_to_start = LatLon.distanceTo(
            LatLon(pos["lat"], pos["lon"], height=pos["alt"] * ft),  # alt ft -> meters
            LatLon(lat_start, lon_start, height=alt_start * ft),  # alt ft -> meters
        )

        if heuristic_name == "distance":
            cost = distance_to_goal

        elif heuristic_name == "fuel":
            # bearing of the plane
            bearing_degrees = aero_bearing(pos["lat"], pos["lon"], lat_to, lon_to)

            # weather computations & A/C speed modification
            we, wn = 0, 0
            temp = 273.15
            if self.weather_interpolator:
                # wind computations
                wind_ms = self.weather_interpolator.interpol_wind_classic(
                    lat=pos["lat"], longi=pos["lon"], alt=pos["alt"], t=pos["ts"]
                )
                we, wn = wind_ms[2][0], wind_ms[2][1]  # 0, 300

                # temperature computations
                temp = self.weather_interpolator.interpol_field(
                    [pos["ts"], pos["alt"], pos["lat"], pos["lon"]], field="T"
                )

                # check for NaN values
                if math.isnan(temp):
                    print("NaN values in temp")

            wspd = sqrt(wn * wn + we * we)

            tas = mach2tas(pos["mach"], pos["alt"] * ft)  # alt ft -> meters

            gs = compute_gspeed(
                tas=tas,
                true_course=radians(bearing_degrees),
                wind_speed=wspd,
                wind_direction=3 * math.pi / 2 - atan2(wn, we),
            )

            # override temp computation
            values_current = {
                "mass": pos["mass"],
                "alt": pos["alt"],
                "speed": tas / kts,
                "temp": temp,
            }

            # compute "time to arrival"
            dt = distance_to_goal / gs

            if distance_to_goal == 0:
                return Value(cost=0)

            if self.perf_model_name == "PS":
                cost = self.perf_model.compute_fuel_consumption(
                    values_current,
                    delta_time=dt,
                    vs=(alt_to - pos["alt"]) * 60 / dt,
                )
            else:
                cost = self.perf_model.compute_fuel_consumption(
                    values_current,
                    delta_time=dt,
                )

        elif heuristic_name == "time":
            we, wn = 0, 0
            bearing_degrees = aero_bearing(pos["lat"], pos["lon"], self.lat2, self.lon2)

            if self.weather_interpolator:
                wind_ms = self.weather_interpolator.interpol_wind_classic(
                    lat=pos["lat"], longi=pos["lon"], alt=pos["alt"], t=pos["ts"]
                )

                we, wn = wind_ms[2][0], wind_ms[2][1]  # 0, 300
            wspd = sqrt(wn * wn + we * we)

            tas = mach2tas(pos["mach"], pos["alt"] * ft)  # alt ft -> meters

            gs = compute_gspeed(
                tas=tas,
                true_course=radians(bearing_degrees),
                wind_speed=wspd,
                wind_direction=3 * math.pi / 2 - atan2(wn, we),
            )

            cost = distance_to_goal / gs

        elif heuristic_name == "lazy_fuel":
            fuel_consummed = s.trajectory.iloc[0]["mass"] - pos["mass"]
            cost = (
                1.05 * distance_to_goal * (fuel_consummed / (distance_to_start + 1e-8))
            )

        elif heuristic_name == "lazy_time":
            cost = (
                1.5
                * distance_to_goal
                * (
                    (pos["ts"] - s.trajectory.iloc[0]["ts"])
                    / (distance_to_start + 1e-8)
                )
            )
        else:
            cost = 0

        return Value(cost=cost)

    def set_network(
        self,
        p0: LatLon,
        p1: LatLon,
        nb_forward_points: int,
        nb_lateral_points: int,
        nb_vertical_points: int,
        climbing_slope: float = None,
        descending_slope: float = None,
        graph_width: float = None,
    ):
        """
        Creation of the airway graph.

        # Parameters
            p0 : Origin of the flight plan
            p1 : Destination of the flight plan
            nb_forward_points (int): Number of forward points in the graph
            nb_lateral_points (int): Number of lateral points in the graph
            nb_vertical_points (int): Number of vertical points in the graph
            climbing_slope (float, optional): Climbing slope of the plane during climbing phase. Defaults to None.
            descending_slope (float, optional):  Descent slope of the plane during descent phase. Defaults to None.
            graph_width (float, optional): Graph width of the graph. Defaults to None.

        # Returns
            A 3D matrix containing for each points its latitude, longitude, altitude between origin & destination.
        """

        cruise_alt_min = 31_000 * ft  # maybe useful to change this
        half_forward_points = nb_forward_points // 2
        half_lateral_points = nb_lateral_points // 2
        half_vertical_points = nb_vertical_points // 2

        distp = (
            graph_width * p0.distanceTo(p1) * 0.022
        )  # meters, around 2.2%*graphwidth of the p0 to p1 distance

        descent_dist = min(
            300_000
            * (
                max(self.ac["cruise"]["height"] - p1.height, 0)
                / self.ac["cruise"]["height"]
            ),
            p0.distanceTo(p1),
        )  # meters

        climb_dist = 220_000 * (
            max(self.ac["cruise"]["height"] - p0.height, 0)
            / self.ac["cruise"]["height"]
        )  # meters

        total_distance = p0.distanceTo(p1)
        if total_distance < (climb_dist + descent_dist):
            climb_dist = total_distance * max(
                (climb_dist / (climb_dist + descent_dist)) - 0.1, 0
            )
            descent_dist = total_distance * max(
                descent_dist / (climb_dist + descent_dist) - 0.1, 0
            )
            possible_altitudes = [cruise_alt_min for k in range(nb_vertical_points)]

        else:
            possible_altitudes = [
                (
                    min(
                        self.ac["cruise"]["height"]
                        + 2000 * ft * i
                        - (self.ac["cruise"]["height"] % 1000),
                        self.ac["limits"]["ceiling"],
                    )
                )
                for i in range(nb_vertical_points)
            ]
        if climbing_slope is not None:
            climbing_ratio = climbing_slope
        else:
            climbing_ratio = (
                possible_altitudes[0] / climb_dist if climb_dist != 0 else 0
            )
        if descending_slope:
            descending_ratio = descending_slope
        else:
            descending_ratio = (
                possible_altitudes[0] / descent_dist if descent_dist != 0 else 0
            )
        # Initialisation of the graph matrix
        pt = [
            [
                [None for k in range(len(possible_altitudes))]
                for j in range(nb_lateral_points)
            ]
            for i in range(nb_forward_points)
        ]

        # set boundaries
        for j in range(nb_lateral_points):
            for k in range(nb_vertical_points):
                pt[0][j][k] = p0
                pt[nb_forward_points - 1][j][k] = p1

        # set climb phase
        i_initial = 1
        if climbing_ratio != 0:
            dist = 0
            alt = p0.height
            while dist < climb_dist and i_initial != nb_forward_points:

                local_dist = (
                    pt[i_initial - 1][half_lateral_points][
                        half_vertical_points
                    ].distanceTo(p1)
                ) / (nb_forward_points - i_initial)
                dist += local_dist
                alt += int(local_dist * climbing_ratio)

                for k in range(nb_vertical_points):
                    bearing = pt[i_initial - 1][half_lateral_points][
                        k
                    ].initialBearingTo(p1)
                    pt[i_initial][half_lateral_points][k] = pt[i_initial - 1][
                        half_lateral_points
                    ][k].destination(
                        local_dist, bearing, min(possible_altitudes[0], alt)
                    )
                i_initial += 1

        # set last step, descent
        i_final = 1
        if descending_ratio != 0:
            dist = 0
            alt = p1.height

            while dist < descent_dist and i_final != nb_forward_points - 1:
                local_dist = (
                    pt[nb_forward_points - i_final][half_lateral_points][
                        half_vertical_points
                    ].distanceTo(p0)
                ) / (nb_forward_points - i_final)
                dist += local_dist
                alt += int(local_dist * descending_ratio)

                for k in range(nb_vertical_points):
                    bearing = pt[nb_forward_points - i_final][half_lateral_points][
                        k
                    ].initialBearingTo(p0)
                    pt[nb_forward_points - i_final - 1][half_lateral_points][k] = pt[
                        nb_forward_points - i_final
                    ][half_lateral_points][k].destination(
                        local_dist, bearing, min(possible_altitudes[0], alt)
                    )
                i_final += 1

        # direct path between end of climbing and beginning of descent
        for k in range(nb_vertical_points):
            for i in range(i_initial, nb_forward_points - i_final + 1):
                bearing = pt[i - 1][half_lateral_points][k].initialBearingTo(p1)
                total_distance = pt[i - 1][half_lateral_points][k].distanceTo(
                    pt[nb_forward_points - 2][half_lateral_points][k]
                )
                pt[i][half_lateral_points][k] = pt[i - 1][half_lateral_points][
                    k
                ].destination(
                    total_distance / (nb_forward_points - i),
                    bearing,
                    height=possible_altitudes[k],
                )

            bearing = pt[half_forward_points - 1][half_lateral_points][
                k
            ].initialBearingTo(pt[half_forward_points + 1][half_lateral_points][k])
            pt[half_forward_points][nb_lateral_points - 1][k] = pt[half_forward_points][
                half_lateral_points
            ][k].destination(
                distp * half_lateral_points,
                bearing + 90,
                height=pt[half_forward_points][half_lateral_points][k].height,
            )
            pt[half_forward_points][0][k] = pt[half_forward_points][
                half_lateral_points
            ][k].destination(
                distp * half_lateral_points,
                bearing - 90,
                height=pt[half_forward_points][half_lateral_points][k].height,
            )

        for j in range(1, half_lateral_points + 1):
            for k in range(len(possible_altitudes)):
                # +j (left)
                bearing = pt[half_forward_points][half_lateral_points + j - 1][
                    k
                ].initialBearingTo(pt[half_forward_points][nb_lateral_points - 1][k])
                total_distance = pt[half_forward_points][half_lateral_points + j - 1][
                    k
                ].distanceTo(pt[half_forward_points][nb_lateral_points - 1][k])
                pt[half_forward_points][half_lateral_points + j][k] = pt[
                    half_forward_points
                ][half_lateral_points + j - 1][k].destination(
                    total_distance / (half_lateral_points - j + 1),
                    bearing,
                    height=pt[half_forward_points][half_lateral_points][k].height,
                )
                # -j (right)
                bearing = pt[half_forward_points][half_lateral_points - j + 1][
                    k
                ].initialBearingTo(pt[half_forward_points][0][k])
                total_distance = pt[half_forward_points][half_lateral_points - j + 1][
                    k
                ].distanceTo(pt[half_forward_points][0][k])
                pt[half_forward_points][half_lateral_points - j][k] = pt[
                    half_forward_points
                ][half_lateral_points - j + 1][k].destination(
                    total_distance / (half_lateral_points - j + 1),
                    bearing,
                    height=pt[half_forward_points][half_lateral_points][k].height,
                )
                for i in range(1, i_initial):
                    alt = pt[i][half_lateral_points][k].height
                    bearing = pt[i - 1][half_lateral_points + j][k].initialBearingTo(
                        pt[half_forward_points][half_lateral_points + j][k]
                    )
                    total_distance = pt[i - 1][half_lateral_points + j][k].distanceTo(
                        pt[half_forward_points][half_lateral_points + j][k]
                    )
                    pt[i][half_lateral_points + j][k] = pt[i - 1][
                        half_lateral_points + j
                    ][k].destination(
                        total_distance / (half_forward_points - i + 1),
                        bearing,
                        height=alt,
                    )

                    bearing = pt[i - 1][half_lateral_points - j][k].initialBearingTo(
                        pt[half_forward_points][half_lateral_points - j][k]
                    )
                    total_distance = pt[i - 1][half_lateral_points - j][k].distanceTo(
                        pt[half_forward_points][half_lateral_points - j][k]
                    )
                    pt[i][half_lateral_points - j][k] = pt[i - 1][
                        half_lateral_points - j
                    ][k].destination(
                        total_distance / (half_forward_points - i + 1),
                        bearing,
                        height=alt,
                    )
                for i in range(i_initial, half_forward_points):
                    # first halp (p0 to np2)
                    bearing = pt[i - 1][half_lateral_points + j][k].initialBearingTo(
                        pt[half_forward_points][half_lateral_points + j][k]
                    )
                    total_distance = pt[i - 1][half_lateral_points + j][k].distanceTo(
                        pt[half_forward_points][half_lateral_points + j][k]
                    )
                    pt[i][half_lateral_points + j][k] = pt[i - 1][
                        half_lateral_points + j
                    ][k].destination(
                        total_distance / (half_forward_points - i + 1),
                        bearing,
                        height=pt[i][half_lateral_points][k].height,
                    )

                    bearing = pt[i - 1][half_lateral_points - j][k].initialBearingTo(
                        pt[half_forward_points][half_lateral_points - j][k]
                    )
                    total_distance = pt[i - 1][half_lateral_points - j][k].distanceTo(
                        pt[half_forward_points][half_lateral_points - j][k]
                    )
                    pt[i][half_lateral_points - j][k] = pt[i - 1][
                        half_lateral_points - j
                    ][k].destination(
                        total_distance / (half_forward_points - i + 1),
                        bearing,
                        height=pt[i][half_lateral_points][k].height,
                    )
                for i in range(1, abs(half_forward_points - i_final)):
                    # second half (np2 to p1)
                    bearing = pt[half_forward_points + i - 1][half_lateral_points + j][
                        k
                    ].initialBearingTo(
                        pt[nb_forward_points - 1][half_lateral_points + j][k]
                    )
                    total_distance = pt[half_forward_points + i - 1][
                        half_lateral_points + j
                    ][k].distanceTo(
                        pt[nb_forward_points - 1][half_lateral_points + j][k]
                    )
                    pt[half_forward_points + i][half_lateral_points + j][k] = pt[
                        half_forward_points + i - 1
                    ][half_lateral_points + j][k].destination(
                        total_distance / (half_forward_points - i + 1),
                        bearing,
                        height=pt[half_forward_points + i][half_lateral_points][
                            k
                        ].height,
                    )

                    bearing = pt[half_forward_points + i - 1][half_lateral_points - j][
                        k
                    ].initialBearingTo(
                        pt[nb_forward_points - 1][half_lateral_points - j][k]
                    )
                    total_distance = pt[half_forward_points + i - 1][
                        half_lateral_points - j
                    ][k].distanceTo(
                        pt[nb_forward_points - 1][half_lateral_points - j][k]
                    )
                    pt[half_forward_points + i][half_lateral_points - j][k] = pt[
                        half_forward_points + i - 1
                    ][half_lateral_points - j][k].destination(
                        total_distance / (half_forward_points - i + 1),
                        bearing,
                        height=pt[half_forward_points + i][half_lateral_points][
                            k
                        ].height,
                    )
                for i in range(abs(half_forward_points - i_final), half_forward_points):
                    alt = pt[half_forward_points + i - 1][half_lateral_points][k].height
                    bearing = pt[half_forward_points + i - 1][half_lateral_points + j][
                        k
                    ].initialBearingTo(
                        pt[nb_forward_points - 1][half_lateral_points + j][k]
                    )
                    total_distance = pt[half_forward_points + i - 1][
                        half_lateral_points + j
                    ][k].distanceTo(
                        pt[nb_forward_points - 1][half_lateral_points + j][k]
                    )
                    pt[half_forward_points + i][half_lateral_points + j][k] = pt[
                        half_forward_points + i - 1
                    ][half_lateral_points + j][k].destination(
                        total_distance / (half_forward_points - i + 1),
                        bearing,
                        height=alt,
                    )

                    bearing = pt[half_forward_points + i - 1][half_lateral_points - j][
                        k
                    ].initialBearingTo(
                        pt[nb_forward_points - 1][half_lateral_points - j][k]
                    )
                    total_distance = pt[half_forward_points + i - 1][
                        half_lateral_points - j
                    ][k].distanceTo(
                        pt[nb_forward_points - 1][half_lateral_points - j][k]
                    )
                    pt[half_forward_points + i][half_lateral_points - j][k] = pt[
                        half_forward_points + i - 1
                    ][half_lateral_points - j][k].destination(
                        total_distance / (half_forward_points - i + 1),
                        bearing,
                        height=alt,
                    )

        return pt

    def get_network(self):
        return self.network

    def flying(
        self, from_: pd.DataFrame, to_: Tuple[float, float, int]
    ) -> pd.DataFrame:
        """Compute the trajectory of a flying object from a given point to a given point

        # Parameters
            from_ (pd.DataFrame): the trajectory of the object so far
            to_ (Tuple[float, float]): the destination of the object

        # Returns
            pd.DataFrame: the final trajectory of the object
        """
        pos = from_.to_dict("records")[0]

        lat_to, lon_to, alt_to = to_[0], to_[1], to_[2]
        dist_ = distance(
            pos["lat"], pos["lon"], lat_to, lon_to, h=(alt_to - pos["alt"]) * ft
        )

        data = []
        epsilon = 100
        dt = 600
        dist = dist_
        loop = 0

        while dist > epsilon:
            # bearing of the plane
            bearing_degrees = aero_bearing(pos["lat"], pos["lon"], lat_to, lon_to)

            # wind computations & A/C speed modification
            we, wn = 0, 0
            temp = 273.15
            if self.weather_interpolator:
                time = pos["ts"] % (3_600 * 24)

                # wind computations
                wind_ms = self.weather_interpolator.interpol_wind_classic(
                    lat=pos["lat"], longi=pos["lon"], alt=alt_to, t=time
                )
                we, wn = wind_ms[2][0], wind_ms[2][1]

                # temperature computations
                temp = self.weather_interpolator.interpol_field(
                    [pos["ts"], pos["alt"], pos["lat"], pos["lon"]], field="T"
                )

            wspd = sqrt(wn * wn + we * we)

            tas = mach2tas(self.mach, alt_to * ft)  # alt ft -> meters

            gs = compute_gspeed(
                tas=tas,
                true_course=radians(bearing_degrees),
                wind_speed=wspd,
                wind_direction=3 * math.pi / 2 - atan2(wn, we),
            )

            if gs * dt > dist:
                # Last step. make sure we go to destination.
                dt = dist / gs
                ll = lat_to, lon_to
            else:
                ll = latlon(
                    pos["lat"],
                    pos["lon"],
                    d=gs * dt,
                    brg=bearing_degrees,
                    h=alt_to * ft,
                )

            values_current = {
                "mass": pos["mass"],
                "alt": pos["alt"],
                "speed": tas / kts,
                "temp": temp,
            }

            pos["fuel"] = self.perf_model.compute_fuel_consumption(
                values_current,
                delta_time=dt,
                vs=(alt_to - pos["alt"]) * 60 / dt,  # ft/min
                # approximation for small angles: tan(alpha) ~ alpha
            )

            mass = pos["mass"] - pos["fuel"]

            # get new weather interpolators
            if pos["ts"] + dt >= (3_600.0 * 24.0):
                if self.weather_date:
                    if self.weather_date == self.initial_date:
                        self.weather_date = self.weather_date.next_day()
                        self.weather_interpolator = self.get_weather_interpolator()
            else:
                if self.weather_date != self.initial_date:
                    self.weather_date = self.weather_date.previous_day()
                    self.weather_interpolator = self.get_weather_interpolator()

            new_row = {
                "ts": (pos["ts"] + dt),
                "lat": ll[0],
                "lon": ll[1],
                "mass": mass,
                "mach": self.mach,
                "fuel": pos["fuel"],
                "alt": alt_to,  # to be modified
            }

            dist = distance(
                ll[0],
                ll[1],
                lat_to,
                lon_to,
                h=(pos["alt"] - alt_to) * ft,  # height difference in m
            )

            if dist < dist_:
                data.append(new_row)
                dist_ = dist
                pos = data[-1]

            else:
                dt = int(dt / 10)
                print("going in the wrong part.")
                assert dt > 0

            loop += 1

        return pd.DataFrame(data)

    def get_weather_interpolator(self) -> GenericWindInterpolator:
        weather_interpolator = None

        if self.weather_date:
            w_dict = self.weather_date.to_dict()

            mat = get_weather_matrix(
                year=w_dict["year"],
                month=w_dict["month"],
                day=w_dict["day"],
                forecast=w_dict["forecast"],
                delete_npz_from_local=False,
                delete_grib_from_local=False,
            )

            # returns both wind and temperature interpolators
            weather_interpolator = GenericWindInterpolator(file_npz=mat)

        return weather_interpolator

    def custom_rollout(self, solver, max_steps=100, make_img=True):
        observation = self.reset()

        solver.reset()
        clear_output(wait=True)

        # loop until max_steps or goal is reached
        for i_step in range(1, max_steps + 1):

            # choose action according to solver
            action = solver.sample_action(observation)

            # get corresponding action
            outcome = self.step(action)
            observation = outcome.observation

            # self.observation = observation

            print("step ", i_step)
            print("policy = ", action[0], action[1])
            print("New state = ", observation.pos)
            print("Alt = ", observation.alt)
            print("Mach = ", observation.trajectory.iloc[-1]["mach"])
            print(observation)

            # if make_img:
            #     # update image
            #     plt.clf()  # clear figure
            #     clear_output(wait=True)
            #     figure = self.render(observation)
            #     # plt.savefig(f'step_{i_step}')

            # final state reached?
            if self.is_terminal(observation):
                break
        if make_img:
            print("Final state reached")
            clear_output(wait=True)
            fig = plot_full(self, observation.trajectory)
            # plt.savefig(f"full_plot")
            plt.show()
            pass
            # clear_output(wait=True)
            # plt.title(f'Flight plan - {self.origin} -> {self.destination} \n Model: {self.perf_model_name}, Fuel: {np.round(observation.trajectory["fuel"].sum(), 2)} Kg')
            # plt.savefig(f"terminal")
            # plt.show()

            # figure = plot_altitude(observation.trajectory)
            # plt.savefig("altitude")
            # plt.show()
            # figure = plot_mass(observation.trajectory)
            # plt.savefig("mass")
            # plt.show()
            # plot_network(self)
        # goal reached?
        is_goal_reached = self.is_goal(observation)

        terminal_state_constraints = self._get_terminal_state_time_fuel(observation)
        if is_goal_reached:
            if self.constraints is not None:
                if self.constraints["time"] is not None:
                    if (
                        self.constraints["time"][1]
                        >= terminal_state_constraints["time"]
                    ):
                        if (
                            self.constraints["fuel"]
                            >= terminal_state_constraints["fuel"]
                        ):
                            print(f"Goal reached after {i_step} steps!")
                        else:
                            print(
                                f"Goal reached after {i_step} steps, but there is not enough fuel remaining!"
                            )
                    else:
                        print(
                            f"Goal reached after {i_step} steps, but not in the good timelapse!"
                        )
                else:
                    if self.constraints["fuel"] >= terminal_state_constraints["fuel"]:
                        print(f"Goal reached after {i_step} steps!")
                    else:
                        print(
                            f"Goal reached after {i_step} steps, but there is not enough fuel remaining!"
                        )
            else:
                if self.ac["limits"]["MFC"] >= terminal_state_constraints["fuel"]:
                    print(f"Goal reached after {i_step} steps!")
                else:
                    print(
                        f"Goal reached after {i_step} steps, but there is not enough fuel remaining!"
                    )
        else:
            print(f"Goal not reached after {i_step} steps!")

        return terminal_state_constraints, self.constraints


def compute_gspeed(
    tas: float, true_course: float, wind_speed: float, wind_direction: float
):
    # Tas : speed in m/s
    # course : current bearing
    # wind speed, wind norm in m/s
    # wind_direction : (3pi/2-arctan(north_component/east_component)) in radian
    ws = wind_speed
    wd = wind_direction
    tc = true_course

    # calculate wind correction angle wca and ground speed gs
    swc = ws / tas * sin(wd - tc)
    if abs(swc) >= 1.0:
        # Wind is to strong
        gs = tas
        error = "Wind is too strong"
    else:
        wca = asin(swc)  # * 180.0 / pi)
        gs = tas * sqrt(1 - swc * swc) - ws * cos(wd - tc)

    if gs < 0:
        # Wind is to strong
        gs = tas
        error = "Wind is too strong"
    else:
        # Reset possible status message
        error = ""
    return gs


def fuel_optimisation(
    origin: Union[str, tuple],
    destination: Union[str, tuple],
    actype: str,
    constraints: dict,
    weather_date: WeatherDate,
    solver_cls: Type[Solver],
    solver_kwargs: Dict[str, Any],
    max_steps: int = 100,
    fuel_tol: float = 1e-3,
) -> float:
    """
    Function to optimise the fuel loaded in the plane, doing multiple fuel loops to approach an optimal

    # Parameters
        origin (Union[str, tuple]):
            ICAO code of the departure airport of th flight plan e.g LFPG for Paris-CDG, or a tuple (lat,lon)

        destination (Union[str, tuple]):
            ICAO code of the arrival airport of th flight plan e.g LFBO for Toulouse-Blagnac airport, or a tuple (lat,lon)

        actype (str):
            Aircarft type describe in openap datas (https://github.com/junzis/openap/tree/master/openap/data/aircraft)

        constraints (dict):
            Constraints that will be defined for the flight plan

        wind_interpolator (GenericWindInterpolator):
            Define the wind interpolator to use wind informations for the flight plan

        fuel_loaded (float):
            Fuel loaded in the plane for the flight

        solver_cls (type[Solver]):
            Solver class used in the fuel loop.

        solver_kwargs (Dict[str, Any]):
            Kwargs to initialize the solver used in the fuel loop.

        max_steps (int):
            max steps to use in the internal fuel loop

        fuel_tol (float):
            tolerance on fuel used to stop the optimization

    # Returns
        float:
            Return the quantity of fuel to be loaded in the plane for the flight
    """

    small_diff = False
    step = 0
    new_fuel = constraints["fuel"]
    while not small_diff:
        domain_factory = lambda: FlightPlanningDomain(
            origin=origin,
            destination=destination,
            actype=actype,
            constraints=constraints,
            weather_date=weather_date,
            objective="distance",
            heuristic_name="distance",
            nb_forward_points=41,
            nb_lateral_points=11,
            fuel_loaded=new_fuel,
            starting_time=0.0,
        )
        solver_kwargs = dict(solver_kwargs)
        solver_kwargs["domain_factory"] = domain_factory
        solver_factory = lambda: solver_cls(**solver_kwargs)
        fuel_prec = new_fuel
        new_fuel = simple_fuel_loop(
            solver_factory=solver_factory,
            domain_factory=domain_factory,
            max_steps=max_steps,
        )
        step += 1
        small_diff = (fuel_prec - new_fuel) <= fuel_tol

    return new_fuel


def simple_fuel_loop(solver_factory, domain_factory, max_steps: int = 100) -> float:
    domain = domain_factory()
    with solver_factory() as solver:
        solver.solve()
        observation: State = domain.reset()
        solver.reset()

        # loop until max_steps or goal is reached
        for i_step in range(1, max_steps + 1):

            # choose action according to solver
            action = solver.sample_action(observation)

            # get corresponding action
            outcome = domain.step(action)
            observation = outcome.observation

            if domain.is_terminal(observation):
                break

        # Retrieve fuel for the flight
        fuel = domain._get_terminal_state_time_fuel(observation)["fuel"]

    return fuel
