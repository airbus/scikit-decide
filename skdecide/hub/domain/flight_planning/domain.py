import math
from argparse import Action
from enum import Enum

# data and math
from math import asin, atan2, cos, radians, sin, sqrt

# typing
from typing import Any, Dict, List, Optional, Tuple, Type, Union

# plotting
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

# aircraft performance model
from openap.extra.aero import bearing as aero_bearing
from openap.extra.aero import distance, ft, latlon, mach2cas, mach2tas
from openap.extra.nav import airport
from pygeodesy.ellipsoidalVincenty import LatLon

from skdecide import DeterministicPlanningDomain, ImplicitSpace, Solver, Space, Value
from skdecide.builders.domain import Renderable, UnrestrictedActions
from skdecide.hub.domain.flight_planning.aircraft_performance.bean.aircraft_state import (
    AircraftState,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.bean.atmos_isa import (
    temperature,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.performance.aero.service.aerodynamics_service import (
    AerodynamicsService,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.performance.phase_enum import (
    PhaseEnum,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.performance.propulsion.service.propulsion_service import (
    PropulsionService,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.performance.rating_enum import (
    RatingEnum,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.weather.service.atmosphere_service import (
    AtmosphereService,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.weather.settings.isa_atmosphere_settings import (
    IsaAtmosphereSettings,
)
from skdecide.hub.domain.flight_planning.flightplanning_utils import plot_full
from skdecide.hub.domain.flight_planning.graph import create_flight_graph, prune_graph
from skdecide.hub.domain.flight_planning.weather_interpolator.weather_tools.get_weather_noaa import (
    get_weather_matrix,
)
from skdecide.hub.domain.flight_planning.weather_interpolator.weather_tools.interpolator.GenericInterpolator import (
    GenericWindInterpolator,
)
from skdecide.hub.domain.flight_planning.weather_interpolator.weather_tools.interpolator.WeatherInterpolator import (
    WeatherForecastInterpolator,
)
from skdecide.hub.space.gym import DiscreteSpace, EnumSpace, ListSpace, TupleSpace


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
    id: Dict[str, float]

    def __init__(self, trajectory, id):
        """Initialisation of a state

        Args:
            trajectory : Trajectory information of the flight
            id: Node id in the airway graph
        """
        self.trajectory = trajectory
        self.id = id

        if trajectory is not None:
            self.mass = trajectory.iloc[-1]["mass"]
            self.alt = trajectory.iloc[-1]["alt"]
            self.time = trajectory.iloc[-1]["ts"]
        else:
            self.mass = None
            self.alt = None
            self.time = None

    def __hash__(self):
        return hash((self.id, int(self.mass), self.alt, int(self.time)))

    def __eq__(self, other):
        return (
            self.id == other.id
            and int(self.mass) == int(other.mass)
            and self.alt == other.alt
            and int(self.time) == int(other.time)
        )

    def __ne__(self, other):
        return (
            self.id != other.id
            or int(self.mass) != int(other.mass)
            or self.alt != other.alt
            or int(self.time) != int(other.time)
        )

    def __str__(self):
        return f"[{self.trajectory.iloc[-1]['ts']:.2f} \
            {self.id} \
            {self.trajectory.iloc[-1]['alt']:.2f} \
            {self.trajectory.iloc[-1]['mass']:.2f}]"


class H_Action(Enum):
    """
    Horizontal action that can be perform by the aircraft
    """

    left = -1
    straight = 0
    right = 1


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
        aircraft_state: AircraftState,
        mach_cruise: float = 0.78,
        mach_climb: float = 0.7,
        mach_descent: float = 0.6,
        nb_forward_points: int = 30,
        nb_lateral_points: int = 10,
        nb_climb_descent_steps: int = 4,
        flight_levels_ft: List[int] = [36_000, 38_000],
        graph_width: str = "medium",
        origin: Optional[Tuple[str, LatLon]] = None,
        destination: Optional[Tuple[str, LatLon]] = None,
        objective: str = "fuel",
        heuristic_name: str = "fuel",
        starting_time: float = 3_600.0 * 8.0,
        weather_date: Optional[WeatherDate] = None,
        wind_interpolator: Optional[GenericWindInterpolator] = None,
        constraints=None,
        fuel_loaded: Optional[float] = None,
        fuel_loop: bool = False,
        fuel_loop_solver_cls: Optional[Type[Solver]] = None,
        fuel_loop_solver_kwargs: Optional[Dict[str, Any]] = None,
        fuel_loop_tol: float = 1e-3,
        res_img_dir: Optional[str] = None,
    ):
        """Initialisation of a flight planning instance

        # Parameters
            origin (Union[str, tuple]):
                ICAO code of the airport, or a tuple (lat,lon,alt), of the origin of the flight plan. Altitude should be in ft
            destination (Union[str, tuple]):
                ICAO code of the airport, or a tuple (lat,lon,alt), of the destination of the flight plan. Altitude should be in ft
            aircraft_state (AircraftState)
                Initial aircraft state.
            cruise_height_min (float)
                Minimum cruise height in ft
            cruise_height_max (float)
                Maximum cruise height in ft
            weather_date (WeatherDate, optional):
                Date for the weather, needed for days management.
                If None, no wind will be applied.
            objective (str, optional):
                Cost function of the flight plan. It can be either fuel, distance or time. Defaults to "fuel".
            heuristic_name (str, optional):
                Heuristic of the flight plan, it will guide the aircraft through the graph. It can be either fuel, distance or time. Defaults to "fuel".
            wind_interpolator (GenericWindInterpolator, optional):
                Wind interpolator for the flight plan. If None, create one from the specified weather_date.
                The data is either already present locally or be downloaded from https://www.ncei.noaa.gov
            constraints (_type_, optional):
                Constraints dictionnary (keyValues : ['time', 'fuel'] ) to be defined in for the flight plan. Defaults to None.
            nb_forward_points (int, optional):
                Number of forward nodes in the graph. Defaults to 41.
            nb_lateral_points (int, optional):
                Number of lateral nodes in the graph. Defaults to 11.
            nb_vertical_points (int, optional):
                Number of vertical nodes in the graph. Defaults to None.
            fuel_loaded (float, optional):
                Fuel loaded in the airscraft for the flight plan. Defaults to None.
            fuel_loop (bool, optional):
                Boolean to create a fuel loop to optimize the fuel loaded for the flight. Defaults to False
            fuel_loop_solver_cls (type[Solver], optional):
                Solver class used in the fuel loop. Defaults to LazyAstar.
            fuel_loop_solver_kwargs (Dict[str, Any], optional):
                Kwargs to initialize the solver used in the fuel loop.
            graph_width (str, optional):
                Airways graph width, in ["small", "medium", "large", "xlarge"]. Defaults to None
            res_img_dir (str, optional):
                Directory in which images will be saved. Defaults to None
            starting_time (float, optional):
                Start time of the flight, in seconds. Defaults to 8AM (3_600.0 * 8.0)
        """
        # Aircraft
        self.initial_aircraft_state = aircraft_state
        self.mach_cruise = mach_cruise
        self.mach_climb = mach_climb
        self.mach_descent = mach_descent

        # Graph construction
        self.nb_forward_points = nb_forward_points
        self.nb_lateral_points = nb_lateral_points
        self.nb_climb_descent_steps = nb_climb_descent_steps
        self.flight_levels_ft = flight_levels_ft
        self.graph_width = graph_width
        self.origin = origin
        self.destination = destination

        # Graph solving
        self.objective = objective
        self.heuristic_name = heuristic_name
        self.starting_time = starting_time

        # Fuel loop
        self.constraints = constraints
        self.fuel_loaded = fuel_loaded
        self.fuel_loop = fuel_loop
        self.fuel_loop_solver_cls = fuel_loop_solver_cls
        self.fuel_loop_solver_kwargs = fuel_loop_solver_kwargs
        self.fuel_loop_tol = fuel_loop_tol

        # Other
        self.res_img_dir = res_img_dir

        # Aircraft Performance objects
        self.atmosphere_service = AtmosphereService()
        self.propulsion_service = PropulsionService()
        self.aerodynamics_service = AerodynamicsService()

        self.propulsion_settings = self.propulsion_service.init_settings(
            model_path=aircraft_state.model_type,
            performance_model_type=aircraft_state.performance_model_type,
        )
        self.aerodynamics_settings = self.aerodynamics_service.init_settings(
            model_path=aircraft_state.model_type,
            performance_model_type=aircraft_state.performance_model_type,
        )

        # --- Initialize the network ---
        if isinstance(origin, str):  # origin is an airport
            ap1 = airport(origin)
            self.origin = LatLon(ap1["lat"], ap1["lon"], ap1["alt"] * ft)
            self.lat1 = ap1["lat"]
            self.lon1 = ap1["lon"]
            self.alt1 = ap1["alt"]
        else:
            self.lat1 = origin.lat
            self.lon1 = origin.lon
            self.alt1 = origin.height / ft

        if isinstance(destination, str):
            ap2 = airport(destination)
            self.destination = LatLon(ap2["lat"], ap2["lon"], ap2["alt"] * ft)
            self.lat2 = ap2["lat"]
            self.lon2 = ap2["lon"]
            self.alt2 = ap2["alt"]
        else:
            self.lat2 = destination.lat
            self.lon2 = destination.lon
            self.alt2 = destination.height / ft

        # Weather
        self.weather_date = weather_date
        self.initial_date = weather_date
        if wind_interpolator is None:
            self.weather_interpolator = self.get_weather_interpolator()

        self.network = self.set_network(
            p0=self.origin,
            p1=self.destination,
            nb_forward_points=nb_forward_points,
            nb_lateral_points=nb_lateral_points,
            nb_climb_descent_steps=nb_climb_descent_steps,
            flight_levels_ft=flight_levels_ft,
            graph_width=graph_width,
        )

        self.node_id_to_key = {
            i: node_key for i, node_key in enumerate(self.network.nodes())
        }
        self.node_key_to_id = {
            node_key: i for i, node_key in enumerate(self.network.nodes())
        }

        # self.fuel_loaded = fuel_loaded

        # # Initialisation of the flight plan, with the initial state
        # if fuel_loop:
        #     if fuel_loop_solver_cls is None:
        #         LazyAstar = load_registered_solver("LazyAstar")
        #         fuel_loop_solver_cls = LazyAstar
        #         fuel_loop_solver_kwargs = dict(heuristic=lambda d, s: d.heuristic(s))
        #     elif fuel_loop_solver_kwargs is None:
        #         fuel_loop_solver_kwargs = {}
        #     fuel_loaded = fuel_optimisation(
        #         origin=origin,
        #         destination=destination,
        #         aircraft_state=self.aircraft_state,
        #         cruise_height_min=cruise_height_min,
        #         cruise_height_max=cruise_height_max,
        #         constraints=constraints,
        #         weather_date=weather_date,
        #         solver_cls=fuel_loop_solver_cls,
        #         solver_kwargs=fuel_loop_solver_kwargs,
        #         fuel_tol=fuel_loop_tol,
        #     )
        #     # Adding fuel reserve (but we can't put more fuel than maxFuel)
        #     fuel_loaded = min(1.1 * fuel_loaded, aircraft_state.MFC)
        # elif fuel_loaded:
        #     self.constraints["fuel"] = (
        #         0.97 * fuel_loaded
        #     )  # Update of the maximum fuel there is to be used
        # else:
        #     fuel_loaded = aircraft_state.MFC

        # self.fuel_loaded = fuel_loaded

        # assert fuel_loaded <= aircraft_state.MFC  # Ensure fuel loaded <= fuel capacity

        id = 0
        if (
            self.initial_aircraft_state.x_graph is not None
            and self.initial_aircraft_state.y_graph is not None
            and self.initial_aircraft_state.z_graph is not None
        ):
            starting_node = self.network.nodes[
                (
                    self.initial_aircraft_state.x_graph,
                    self.initial_aircraft_state.y_graph,
                    self.initial_aircraft_state.z_graph,
                )
            ]
            self.origin: LatLon = starting_node["latlon"]
            self.origin.height = starting_node["flight_level"] * ft
            self.lat1 = self.origin.lat
            self.lon1 = self.origin.lon
            self.alt1 = self.origin.height / ft
            id = self.node_key_to_id[
                (
                    self.initial_aircraft_state.x_graph,
                    self.initial_aircraft_state.y_graph,
                    self.initial_aircraft_state.z_graph,
                )
            ]
            print(
                f"Aircraft zp: {self.initial_aircraft_state.zp_ft} does not match its initial position altitude: {self.alt1}"
            )
            print("Setting aircraft zp to node altitude")
            self.initial_aircraft_state.zp_ft = self.alt1

        self.start = State(
            trajectory=pd.DataFrame(
                [
                    {
                        "ts": self.starting_time,
                        "lat": self.origin.lat,
                        "lon": self.origin.lon,
                        "alt": self.origin.height / ft,
                        "mass": aircraft_state.gw_kg,
                        "mach": aircraft_state.mach,
                        "cas": mach2cas(
                            mach=aircraft_state.mach, h=self.origin.height, dT=0
                        ),
                        "fuel": 0.0,
                        "phase": "climb",
                    }
                ]
            ),
            id=id,
        )

        # self.res_img_dir = res_img_dir

    # Class functions
    def _get_next_state(self, memory: D.T_state, action: D.T_event) -> D.T_state:
        """
        Compute the next state, adapted to work with the structure of the generated flight graph.

        Args:
            memory (D.T_state): The current state.
            action (D.T_event): The action to perform.

        Returns:
            D.T_state: The next state, or the current state if the action is not possible.
        """

        memory_key = self.node_id_to_key[memory.id]
        current_y = memory_key[1]
        current_z = memory_key[2]

        node_successors = np.array(list(self.network.successors(memory_key)))
        successors_y = node_successors[:, 1]
        successors_z = node_successors[:, 2]

        if node_successors.shape == 0:
            return memory

        horizontal_indices = None
        if action[0].name == "straight":
            horizontal_indices = np.where(successors_y == current_y)[0]
        elif action[0].name == "left":
            horizontal_indices = np.where(successors_y < current_y)[0]
        else:
            horizontal_indices = np.where(successors_y > current_y)[0]

        vertical_indices = None
        if action[1].name == "cruise":
            vertical_indices = np.where(successors_z == current_z)[0]
        elif action[1].name == "descent":
            vertical_indices = np.where(successors_z < current_z)[0]
        else:
            vertical_indices = np.where(successors_z > current_z)[0]

        if horizontal_indices is not None and vertical_indices is not None:
            index = np.intersect1d(horizontal_indices, vertical_indices)
            next_node_key = tuple(node_successors[index][0])
            next_node = self.network.nodes[next_node_key]
        else:
            return memory

        p: LatLon = next_node["latlon"]
        to_lat = p.lat
        to_lon = p.lon
        to_alt = next_node["flight_level"]
        current_phase = next_node["phase"]

        if current_phase == "cruise":
            current_phase = PhaseEnum.CRUISE
            current_rating = RatingEnum.CR
            current_speed = self.mach_cruise
        elif current_phase == "climb":
            current_phase = PhaseEnum.CLIMB
            current_rating = RatingEnum.MCL
            current_speed = self.mach_climb
        elif current_phase == "descent":
            current_phase = PhaseEnum.DESCENT
            current_rating = RatingEnum.IDLE
            current_speed = self.mach_descent
        elif current_phase == "end":
            current_phase = PhaseEnum.DESCENT
            current_rating = RatingEnum.IDLE
            current_speed = self.mach_descent
        else:
            raise ValueError("Current phase key not recognized.")

        new_trajectory_segment = self.flying(
            from_=memory.trajectory.tail(1),
            to_=(to_lat, to_lon, to_alt),  # Pass altitude in meters
            current_speed=current_speed,
            current_phase=current_phase,
            current_rating=current_rating,
        )

        state = State(
            pd.concat([memory.trajectory, new_trajectory_segment], ignore_index=True),
            self.node_key_to_id[next_node_key],
        )
        return state

    def _get_transition_value(
        self,
        memory: D.T_state,
        action: D.T_event,
        next_state: D.T_state,
    ) -> Value[D.T_value]:
        """
        Get the value (reward or cost) of a transition.
        Set cost to distance travelled between points

        Args:
            memory (D.T_state): The current state
            action (D.T_event): The action to perform
            next_state (Optional[D.T_state], optional): The next state. Defaults to None.

        Returns:
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
        end_node_key = list(self.network.nodes)[-1]
        end_node_id = self.node_key_to_id[end_node_key]
        # print(ImplicitSpace(lambda x: x.id == end_node_id).contains())

        return ImplicitSpace(lambda x: x.id == end_node_id)

    def _get_terminal_state_time_fuel(self, state: State) -> dict:
        """
        Get the domain terminal state information to compare with the constraints

        Args:
            state (State): terminal state to retrieve the information on fuel and time.

        Returns:
            dict: dictionnary containing both fuel and time information.
        """
        fuel = 0.0
        for trajectory in state.trajectory.iloc:
            fuel += trajectory["fuel"]

        if (
            state.trajectory.iloc[-1]["ts"] < self.starting_time
        ):  # The flight arrives the next day
            time = 3_600 * 24 - self.starting_time + state.trajectory.iloc[-1]["ts"]
        else:
            time = state.trajectory.iloc[-1]["ts"] - self.starting_time

        return {"time": time, "fuel": fuel}

    def _is_terminal(self, state: State) -> D.T_predicate:
        """
        Indicate whether a state is terminal.

        Stop an episode only when goal reached.
        """

        # The state is terminal if it does not have any successors
        return state.id == len(list(self.network.nodes)) - 1

    def _get_applicable_actions_from(self, state: State) -> Space[D.T_event]:
        """
        Get the applicable actions from a state.
        """
        current_node_id = state.id

        current_node_key = self.node_id_to_key[current_node_id]
        current_y = current_node_key[1]
        current_z = current_node_key[2]

        successor_node_keys = list(self.network.successors(current_node_key))
        successor_node_keys.sort()

        # no successors => no actions
        if not successor_node_keys:
            return ListSpace([])

        space = []
        for successor in successor_node_keys:
            successor_y = successor[1]
            successor_z = successor[2]

            if successor_z == current_z:
                vertical_action = V_Action.cruise
            elif successor_z > current_z:
                vertical_action = V_Action.climb
            else:
                vertical_action = V_Action.descent

            if successor_y == current_y:
                horizontal_action = H_Action.straight
            elif successor_y > current_y:
                horizontal_action = H_Action.right
            else:
                horizontal_action = H_Action.left

            space.append((horizontal_action, vertical_action))

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

        return TupleSpace(
            (
                DiscreteSpace(self.network.number_of_nodes()),
                DiscreteSpace(self.network.number_of_nodes()),
            )
        )

    def _render_from(self, memory: State, **kwargs: Any) -> Any:
        """
        Render visually the map.

        Returns:
            matplotlib figure
        """
        return
        # return plot_trajectory(
        #     self.lat1,
        #     self.lon1,
        #     self.lat2,
        #     self.lon2,
        #     memory.trajectory,
        # )

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
            if pos["phase"] == "cruise":
                current_phase = PhaseEnum.CRUISE
                current_rating = RatingEnum.CR
                current_speed = self.mach_cruise
            elif pos["phase"] == "climb":
                current_phase = PhaseEnum.CLIMB
                current_rating = RatingEnum.MCL
                current_speed = self.mach_climb
            elif pos["phase"] == "descent":
                current_phase = PhaseEnum.DESCENT
                current_rating = RatingEnum.IDLE
                current_speed = self.mach_descent
            else:
                raise ValueError("Current phase key not recognized.")

            aircraft_state = AircraftState(
                model_type=self.initial_aircraft_state.model_type,
                performance_model_type=self.initial_aircraft_state.performance_model_type,
                gw_kg=pos["mass"],
                zp_ft=pos["alt"],
                mach=current_speed,
                phase=current_phase,
                rating_level=current_rating,
                cg=0.3,
                gamma_air_deg=0,
            )
            if current_phase.name == "CRUISE":
                # compute the thrust coefficient for cruise
                if alt_to - aircraft_state.zp_ft > 0:
                    reduction_coeff = 1.25
                elif alt_to - aircraft_state.zp_ft == 0:
                    reduction_coeff = 1.0
                else:
                    reduction_coeff = 0.75

                cx = self.aerodynamics_service.compute_drag_coefficient(
                    aerodynamics_settings=self.aerodynamics_settings,
                    aircraft_state=aircraft_state,
                )
                aircraft_state.cx = cx
                dynamic_pressure = (
                    0.7
                    * aircraft_state.weather_state.static_pressure_pa
                    * aircraft_state.mach**2
                )
                aircraft_state.drag_n = (
                    aircraft_state.cx * dynamic_pressure * self.propulsion_settings.sref
                )
                aircraft_state.thrust_n = aircraft_state.drag_n * reduction_coeff
                try:
                    aircraft_state.tsp = (
                        self.propulsion_service.compute_tsp_from_thrust(
                            propulsion_settings=self.propulsion_settings,
                            aircraft_state=aircraft_state,
                            target_thrust_n=aircraft_state.thrust_n,
                        )
                    )
                except:
                    return np.inf
            elif current_phase.name == "CLIMB":
                aircraft_state.tsp = self.propulsion_service.compute_max_rating(
                    propulsion_settings=self.propulsion_settings,
                    aircraft_state=aircraft_state,
                )
            elif current_phase.name == "DESCENT":
                aircraft_state.tsp = 0.0
            else:
                raise ValueError("Current phase not recognized in loop.")

            aircraft_state.cl = (2 * aircraft_state.gw_kg * 9.80665) / (
                delta
                * 101325.0
                * 1.4
                * self.propulsion_settings.sref
                * aircraft_state.mach**2
            )

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
                    [pos["ts"], (pos["alt"] + alt_to) / 2, pos["lat"], pos["lon"]],
                    field="temperature",
                )

                # check for NaN values
                if math.isnan(temp):
                    print("NaN values in temp")

            # compute dISA
            dISA = temp - temperature((pos["alt"] + alt_to) / 2, disa=0)
            isa_atmosphere_settings = IsaAtmosphereSettings(d_isa=dISA)

            weather_state = self.atmosphere_service.retrieve_weather_state(
                atmosphere_settings=isa_atmosphere_settings,
                four_dimensions_state=aircraft_state,
            )

            aircraft_state.weather_state = weather_state

            delta = aircraft_state.weather_state.static_pressure_pa / 101325.0
            aircraft_state.cl = (2 * aircraft_state.gw_kg * 9.80665) / (
                delta
                * 101325.0
                * 1.4
                * self.propulsion_settings.sref
                * aircraft_state.mach**2
            )

            wspd = sqrt(wn * wn + we * we)

            tas = mach2tas(
                aircraft_state.mach, aircraft_state.zp_ft * ft
            )  # alt ft -> meters

            aircraft_state.tas_meters_per_sec = tas

            gs = compute_gspeed(
                tas=tas,
                true_course=radians(bearing_degrees),
                wind_speed=wspd,
                wind_direction=3 * math.pi / 2 - atan2(wn, we),
            )

            # compute "time to arrival"
            dt = distance_to_goal / gs

            if distance_to_goal == 0:
                return Value(cost=0)

            ff = self.propulsion_service.compute_total_fuel_flow_kg_per_sec(
                propulsion_settings=self.propulsion_settings,
                aircraft_state=aircraft_state,
            )

            cost = ff * dt

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
        nb_climb_descent_steps: int,
        flight_levels_ft: List[float],
        graph_width: float = "medium",
    ):
        """
        Creation of the airway graph.

        # Parameters
            p0 : Origin of the flight plan
            p1 : Destination of the flight plan
            nb_forward_points (int): Number of forward points in the graph
            nb_lateral_points (int): Number of lateral points in the graph
            nb_climb_descent_steps (int): Number of steps in climb and descent
            flight_levels_ft (List[float]): List of flight levels during cruise
            graph_width (str, optional): small, medium and wide strings (default to medium)

        # Returns
            A 3D matrix containing for each points its latitude, longitude, altitude between origin & destination.
        """

        graph = create_flight_graph(
            p0=p0,
            p1=p1,
            nb_forward_points=nb_forward_points,
            nb_lateral_points=nb_lateral_points,
            nb_climb_descent_steps=nb_climb_descent_steps,
            flight_levels_ft=flight_levels_ft,
            graph_width=graph_width,
        )

        self.network = prune_graph(G=graph)

        return self.network

    def get_network(self):
        return self.network

    def flying(
        self,
        from_: pd.DataFrame,
        to_: Tuple[float, float, int],
        current_speed: float,
        current_phase: PhaseEnum,
        current_rating: RatingEnum,
    ) -> pd.DataFrame:
        """Compute the trajectory of a flying object from a given point to a given point

        # Parameters
            from_ (pd.DataFrame): the trajectory of the object so far
            to_ (Tuple[float, float]): the destination of the object

        # Returns
            pd.DataFrame: the final trajectory of the object
        """
        pos = from_.to_dict("records")[0]

        aircraft_state = AircraftState(
            model_type=self.initial_aircraft_state.model_type,
            performance_model_type=self.initial_aircraft_state.performance_model_type,
            gw_kg=pos["mass"],
            zp_ft=pos["alt"],
            mach=current_speed,
            phase=current_phase,
            rating_level=current_rating,
            cg=0.3,
            gamma_air_deg=0,
        )

        lat_to, lon_to, alt_to = to_[0], to_[1], to_[2]
        dist_ = distance(
            pos["lat"],
            pos["lon"],
            lat_to,
            lon_to,
            h=abs(alt_to - aircraft_state.zp_ft) * ft,
        )

        data = []
        epsilon = 100
        dt = 600
        dist = dist_
        loop = 0

        while dist > epsilon:
            # if dist == 0:
            #     angle = 0
            # else:
            #     angle = atan((alt_to - aircraft_state.zp_ft) * ft / dist)

            # bearing of the plane
            bearing_degrees = aero_bearing(pos["lat"], pos["lon"], lat_to, lon_to)

            # --- WEATHER ---
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
                    [pos["ts"], (pos["alt"] + alt_to) / 2, pos["lat"], pos["lon"]],
                    field="temperature",
                )
                dISA = temp - temperature((pos["alt"] + alt_to) / 2, disa=0)
            else:
                dISA = 0.0

            isa_atmosphere_settings = IsaAtmosphereSettings(d_isa=dISA)
            weather_state = self.atmosphere_service.retrieve_weather_state(
                atmosphere_settings=isa_atmosphere_settings,
                four_dimensions_state=aircraft_state,
            )
            aircraft_state.weather_state = weather_state

            # --- SPEED ---
            tas = mach2tas(aircraft_state.mach, alt_to * ft)  # alt ft -> meters
            aircraft_state.tas_meters_per_sec = tas

            wspd = sqrt(wn * wn + we * we)
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
            aircraft_state.time_step = dt

            delta = aircraft_state.weather_state.static_pressure_pa / 101325.0

            # --- A/C PERFORMANCE
            aircraft_state.cl = (2 * aircraft_state.gw_kg * 9.80665) / (
                delta
                * 101325.0
                * 1.4
                * self.propulsion_settings.sref
                * aircraft_state.mach**2
            )

            if current_phase.name == "CLIMB" or current_phase.name == "CRUISE":
                aircraft_state.tsp = self.propulsion_service.compute_max_rating(
                    propulsion_settings=self.propulsion_settings,
                    aircraft_state=aircraft_state,
                )
            elif current_phase.name == "DESCENT":
                aircraft_state.tsp = 0.0
            else:
                raise ValueError("Current phase not recognized in loop.")

            ff = self.propulsion_service.compute_total_fuel_flow_kg_per_sec(
                propulsion_settings=self.propulsion_settings,
                aircraft_state=aircraft_state,
            )

            pos["fuel"] = ff * dt

            mass_new = aircraft_state.gw_kg - pos["fuel"]

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

            dist = distance(
                ll[0],
                ll[1],
                lat_to,
                lon_to,
                h=abs(aircraft_state.zp_ft - alt_to) * ft,  # height difference in m
            )

            new_row = {
                "ts": (pos["ts"] + dt),
                "lat": ll[0],
                "lon": ll[1],
                "alt": alt_to,
                "mass": mass_new,
                "mach": current_speed,
                "cas": mach2cas(mach=aircraft_state.mach, h=alt_to * ft, dT=0),
                "fuel": pos["fuel"],
                "phase": current_phase.name,
            }

            aircraft_state.gw_kg = mass_new
            aircraft_state.zp_ft = alt_to

            if dist <= dist_:
                data.append(new_row)
                dist_ = dist
                pos = data[-1]

            else:
                dt = int(dt / 10)
                print("going in the wrong part.")
                assert dt > 0

            loop += 1

        return pd.DataFrame(data)

    def get_weather_interpolator(self) -> WeatherForecastInterpolator:
        weather_interpolator = None

        if self.weather_date:
            lat_min = min(self.origin.lat, self.destination.lat) - 1
            lat_max = max(self.origin.lat, self.destination.lat) + 1
            lon_min = min(self.origin.lon, self.destination.lon) - 2
            lon_max = max(self.origin.lon, self.destination.lon) + 2

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
            weather_interpolator = WeatherForecastInterpolator(
                file_npz=mat,
                lat_min=lat_min,
                lat_max=lat_max,
                lon_min=lon_min,
                lon_max=lon_max,
            )

        return weather_interpolator

    def custom_rollout(
        self, solver: Solver, max_steps: int = 100, make_img: bool = True
    ) -> None:

        observation = self.reset()

        # loop until max_steps or goal is reached
        for i_step in range(1, max_steps + 1):

            # choose action according to solver
            action = solver.sample_action(observation)
            outcome = self.step(action)
            observation = outcome.observation

            # final state reached?
            if self.is_terminal(observation):
                break
        if make_img:
            print("Final state reached")
            fig = plot_full(self, observation.trajectory)
            plt.savefig(f"full_plot")
            plt.show()
            pass
        # goal reached?
        is_goal_reached = self.is_goal(observation)

        terminal_state_constraints = self._get_terminal_state_time_fuel(observation)
        if is_goal_reached:
            if self.constraints is not None:
                if self.constraints["time"] is not None:
                    if (
                        self.constraints["time"][1]  # type: ignore
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
                if (
                    self.initial_aircraft_state.MFC
                    >= terminal_state_constraints["fuel"]
                ):
                    print(f"Goal reached after {i_step} steps!")
                else:
                    print(
                        f"Goal reached after {i_step} steps, but there is not enough fuel remaining!"
                    )
        else:
            print(f"Goal not reached after {i_step} steps!")
        self.observation = observation
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
    aircraft_state: AircraftState,
    cruise_height_min: float,
    cruise_height_max: float,
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

        aircraft_state (AircraftState)
            Initial aircraft state.

        cruise_height_min (float)
            Minimum cruise height in ft

        cruise_height_max (float)
            Maximum cruise height in ft

        constraints (dict):
            Constraints that will be defined for the flight plan

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
            aircraft_state=aircraft_state,
            cruise_height_min=cruise_height_min,
            cruise_height_max=cruise_height_max,
            objective="distance",
            heuristic_name="distance",
            constraints=constraints,
            weather_date=weather_date,
            nb_vertical_points=8,
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
