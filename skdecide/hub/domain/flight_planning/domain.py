import math
from argparse import Action
from enum import Enum

# data and math
from math import asin, atan, atan2, cos, radians, sin, sqrt

# typing
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

# plotting
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

# aircraft performance model
from openap.extra.aero import bearing as aero_bearing
from openap.extra.aero import (
    cas2mach,
    cas2tas,
    distance,
    ft,
    kts,
    latlon,
    mach2cas,
    mach2tas,
    nm,
)
from openap.extra.nav import airport
from pygeodesy.ellipsoidalVincenty import LatLon

from skdecide import DeterministicPlanningDomain, ImplicitSpace, Solver, Space, Value
from skdecide.builders.domain import Renderable, UnrestrictedActions
from skdecide.hub.domain.flight_planning.aircraft_performance.bean.aircraft_state import (
    AircraftState,
)
from skdecide.hub.domain.flight_planning.aircraft_performance.bean.atmos_isa import (
    disa_alt_temp,
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
from skdecide.hub.domain.flight_planning.weather_interpolator.weather_tools.interpolator.WeatherInterpolator import (
    WeatherForecastInterpolator,
)
from skdecide.hub.space.gym import DiscreteSpace, EnumSpace, ListSpace, TupleSpace
from skdecide.utils import load_registered_solver


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
        origin: Union[str, tuple],
        destination: Union[str, tuple],
        aircraft_state: AircraftState,
        cruise_height_min: float = 32_000.0,
        cruise_height_max: float = 38_000.0,
        weather_date: Optional[WeatherDate] = None,
        objective: str = "fuel",
        heuristic_name: str = "fuel",
        wind_interpolator: Optional[GenericWindInterpolator] = None,
        constraints=None,
        nb_forward_points: int = 41,
        nb_lateral_points: int = 11,
        nb_vertical_points: Optional[int] = None,
        fuel_loaded: Optional[float] = None,
        fuel_loop: bool = False,
        fuel_loop_solver_cls: Optional[Type[Solver]] = None,
        fuel_loop_solver_kwargs: Optional[Dict[str, Any]] = None,
        fuel_loop_tol: float = 1e-3,
        graph_width: Optional[str] = "medium",
        res_img_dir: Optional[str] = None,
        starting_time: float = 3_600.0 * 8.0,
    ):
        """Initialisation of a flight planning instance

        # Parameters
            origin (Union[str, tuple]):
                ICAO code of the airport, or a tuple (lat,lon,alt), of the origin of the flight plan. Altitude should be in ft
            destination (Union[str, tuple]):
                ICAO code of the airport, or a tuple (lat,lon,alt), of the destination of the flight plan. Altitude should be in ft
            ac_path_name (str):
                Aircraft path or name. In the case of BADA: BADA/ac; in the case of OpenAP or Poll-Schumann: ac
            perf_model_name (PerformanceModelEnum):
                Aircraft performance model used in the flight plan.
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
                Fuel loaded in the airscraft for the flight plan. Defaults to None.
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
                Airways graph width, in ["small", "medium", "large", "xlarge"]. Defaults to None
            res_img_dir (str, optional):
                Directory in which images will be saved. Defaults to None
            starting_time (float, optional):
                Start time of the flight, in seconds. Defaults to 8AM (3_600.0 * 8.0)
        """
        self.aircraft_state = aircraft_state
        self.cruise_height_min = cruise_height_min
        self.cruise_height_max = cruise_height_max

        # other objects for a/c performance
        self.atmosphere_service = AtmosphereService()
        self.propulsion_service = PropulsionService()
        self.aerodynamics_service = AerodynamicsService()

        self.propulsion_settings = self.propulsion_service.init_settings(
            model_path=self.aircraft_state.model_type,
            performance_model_type=self.aircraft_state.performance_model_type,
        )
        self.aerodynamics_settings = self.aerodynamics_service.init_settings(
            model_path=self.aircraft_state.model_type,
            performance_model_type=self.aircraft_state.performance_model_type,
        )

        self.alt_crossover = self.aerodynamics_service.compute_crossover(
            aircraft_state=self.aircraft_state,
            aerodynamics_settings=self.aerodynamics_settings,
        )

        self.take_off_weight = aircraft_state.gw_kg

        # Initialisation of the origin and the destination
        self.origin, self.destination = origin, destination
        if isinstance(origin, str):  # Origin is an airport
            ap1 = airport(origin)
            self.lat1, self.lon1, self.alt1 = (
                ap1["lat"],
                ap1["lon"],
                ap1["alt"],
            )  # altitude in feet
        else:  # Origin is geographic coordinates
            self.lat1, self.lon1, self.alt1 = origin

        if isinstance(destination, str):  # Destination is an airport
            ap2 = airport(destination)
            self.lat2, self.lon2, self.alt2 = (
                ap2["lat"],
                ap2["lon"],
                ap2["alt"],
            )  # altitude in feet
        else:  # Destination is geographic coordinates
            self.lat2, self.lon2, self.alt2 = destination
        self.start_time = starting_time

        self.aircraft_state.mach = cas2mach(
            self.aircraft_state.cas_climb_kts * kts, h=self.alt1 * ft
        )

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

        self.nb_forward_points = nb_forward_points
        self.nb_lateral_points = nb_lateral_points

        if nb_vertical_points:
            self.nb_vertical_points = nb_vertical_points
        else:
            self.nb_vertical_points = (
                int((cruise_height_max - cruise_height_min) / 1000) + 1
            )

        self.network = self.set_network(
            p0=LatLon(self.lat1, self.lon1, self.alt1 * ft),  # alt ft -> meters
            p1=LatLon(self.lat2, self.lon2, self.alt2 * ft),  # alt ft -> meters
            nb_forward_points=self.nb_forward_points,
            nb_lateral_points=self.nb_lateral_points,
            nb_vertical_points=self.nb_vertical_points,
            cas_climb=self.aircraft_state.cas_climb_kts,
            mach_cruise=self.aircraft_state.mach_cruise,
            cas_descent=self.aircraft_state.cas_descent_kts,
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
                aircraft_state=self.aircraft_state,
                constraints=constraints,
                weather_date=weather_date,
                solver_cls=fuel_loop_solver_cls,
                solver_kwargs=fuel_loop_solver_kwargs,
                fuel_tol=fuel_loop_tol,
            )
            # Adding fuel reserve (but we can't put more fuel than maxFuel)
            fuel_loaded = min(1.1 * fuel_loaded, aircraft_state.MFC)
        elif fuel_loaded:
            self.constraints["fuel"] = (
                0.97 * fuel_loaded
            )  # Update of the maximum fuel there is to be used
        else:
            fuel_loaded = aircraft_state.MFC

        self.fuel_loaded = fuel_loaded

        assert fuel_loaded <= aircraft_state.MFC  # Ensure fuel loaded <= fuel capacity

        self.start = State(
            trajectory=pd.DataFrame(
                [
                    {
                        "ts": self.start_time,
                        "lat": self.lat1,
                        "lon": self.lon1,
                        "mass": self.aircraft_state.gw_kg_memory[0]
                        - 0.8 * (self.aircraft_state.MFC - self.fuel_loaded),
                        "cas": self.aircraft_state.cas_climb_kts,
                        "mach": cas2mach(
                            self.aircraft_state.cas_climb_kts * kts, h=self.alt1 * ft
                        ),
                        "speed_type": "CAS",
                        "fuel": 0.0,
                        "alt": self.alt1,
                        "phase": "climb",
                    }
                ]
            ),
            id=0,
        )

        self.res_img_dir = res_img_dir
        isa_atmosphere_settings = IsaAtmosphereSettings(d_isa=0)

        weather_state = self.atmosphere_service.retrieve_weather_state(
            atmosphere_settings=isa_atmosphere_settings,
            four_dimensions_state=self.aircraft_state,
        )

        self.aircraft_state.weather_state = weather_state

        # TODO: depends on PHASE
        self.aircraft_state.tsp = self.propulsion_service.compute_max_rating(
            propulsion_settings=self.propulsion_settings,
            aircraft_state=self.aircraft_state,
        )

    # Class functions
    def _get_next_state(self, memory: D.T_state, action: D.T_event) -> D.T_state:
        """Compute the next state

        Args:
            memory (D.T_state): The current state
            action (D.T_event): The action to perform

        Returns:
            D.T_state: The next state
        """
        trajectory = memory.trajectory.copy()

        # Get current node information
        current_node_id = memory.id
        current_height = self.network.nodes[current_node_id]["height"]
        current_heading = self.network.nodes[current_node_id]["heading"]
        current_phase = self.network.nodes[current_node_id]["phase"]
        current_speed = self.network.nodes[current_node_id]["speed"]
        current_speed_type = self.network.nodes[current_node_id]["speed_type"]

        # Get successors information
        node_successors = list(self.network.successors(current_node_id))
        successors_heights = np.array(
            [self.network.nodes[node_id]["height"] for node_id in node_successors]
        )
        successors_headings = np.array(
            [self.network.nodes[node_if]["heading"] for node_if in node_successors]
        )

        # Horizontal actions
        if action[0].name == "straight":
            index_headings = np.where(successors_headings == current_heading)[0]
        elif action[0].name == "right":
            index_headings = np.where(successors_headings > current_heading)[0]
        elif action[0].name == "left":
            index_headings = np.where(successors_headings < current_heading)[0]
        else:
            raise ValueError("The action is not recognized.")

        # Vertical actions
        if action[1].name == "cruise":
            index_heights = np.where(successors_heights == current_height)[0]
        elif action[1].name == "climb":
            index_heights = np.where(successors_heights > current_height)[0]
        elif action[1].name == "descent":
            index_heights = np.where(successors_heights < current_height)[0]
        else:
            raise ValueError("The action is not recognized.")

        if len(index_headings) == 0 or len(index_heights) == 0:
            return memory

        # Compute the intersection of the indexes to get the next node to reach
        index = np.intersect1d(index_headings, index_heights)
        if len(index) == 0:
            return memory
        else:
            index = index[0]

        # Get the next node information
        next_node = node_successors[index]
        to_lat = self.network.nodes[next_node]["lat"]
        to_lon = self.network.nodes[next_node]["lon"]
        to_alt = self.network.nodes[next_node]["height"] / ft

        # Compute the next trajectory
        trajectory = self.flying(
            trajectory.tail(1),
            (to_lat, to_lon, to_alt),
            current_phase,
            current_speed,
            current_speed_type,
        )

        # Update the next state
        state = State(
            pd.concat([memory.trajectory, trajectory], ignore_index=True),
            next_node,
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
        # print(f"Old ID: {memory.id}, New ID: {next_state.id}")
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
        return ImplicitSpace(lambda x: len(list(self.network.successors(x.id))) == 0)

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
        current_node_id = state.id
        # The state is terminal if it does not have any successors
        return len(list(self.network.successors(current_node_id))) == 0

    def _get_applicable_actions_from(self, memory: D.T_state) -> Space[D.T_event]:
        """
        Get the applicable actions from a state.
        """
        # Get current node information
        current_node_id = memory.id
        current_height = self.network.nodes[current_node_id]["height"]
        current_heading = self.network.nodes[current_node_id]["heading"]

        # Get successors information
        node_successors = list(self.network.successors(current_node_id))
        successors_heights = np.array(
            [self.network.nodes[node_id]["height"] for node_id in node_successors]
        )
        successors_headings = np.array(
            [self.network.nodes[node_if]["heading"] for node_if in node_successors]
        )

        # V_Action
        index_climb = (np.where(successors_heights > current_height)[0], V_Action.climb)
        index_descend = (
            np.where(successors_heights < current_height)[0],
            V_Action.descent,
        )
        index_cruise = (
            np.where(successors_heights == current_height)[0],
            V_Action.cruise,
        )

        # H_Action
        index_straight = (
            np.where(successors_headings == current_heading),
            H_Action.straight,
        )
        index_right = (np.where(successors_headings > current_heading), H_Action.right)
        index_left = (np.where(successors_headings < current_heading), H_Action.left)
        space = []

        for v_actions in [index_climb, index_descend, index_cruise]:
            for h_actions in [index_straight, index_left, index_right]:
                if len(v_actions[0]) > 0 and len(h_actions[0]) > 0:
                    # Compute intersection of the indexes
                    index = np.intersect1d(v_actions[0], h_actions[0])
                    if len(index) > 0:
                        space.append((h_actions[1], v_actions[1]))

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
                DiscreteSpace(self.network.number_of_edges()),
            )
        )

    def _render_from(self, memory: State, **kwargs: Any) -> Any:
        """
        Render visually the map.

        Returns:
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

        self.aircraft_state.update_settings(
            gw_kg=pos["mass"],
            zp_ft=pos["alt"],
        )

        if pos["speed_type"] == "CAS":
            self.aircraft_state.mach = cas2mach(pos["cas"] * kts, h=pos["alt"] * ft)
            if pos["alt"] < self.alt_crossover:
                self.aircraft_state.mach = min(
                    self.aircraft_state.mach, self.aircraft_state.mach_cruise
                )
        else:
            self.aircraft_state.mach = min(pos["mach"], self.aircraft_state.mach_cruise)

        self.aircraft_state.tsp = self.propulsion_service.compute_max_rating(
            propulsion_settings=self.propulsion_settings,
            aircraft_state=self.aircraft_state,
        )

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
            if distance_to_goal != 0:
                # angle of the plane
                angle = atan((alt_to - pos["alt"]) * ft / distance_to_goal)
            else:
                angle = 0

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
                four_dimensions_state=self.aircraft_state,
            )

            self.aircraft_state.weather_state = weather_state

            delta = self.aircraft_state.weather_state.static_pressure_pa / 101325.0
            self.aircraft_state.cl = (2 * self.aircraft_state.gw_kg * 9.80665) / (
                delta
                * 101325.0
                * 1.4
                * self.propulsion_settings.sref
                * self.aircraft_state.mach**2
                * math.cos(angle)
            )

            wspd = sqrt(wn * wn + we * we)

            tas = mach2tas(
                self.aircraft_state.mach, self.aircraft_state.zp_ft * ft
            )  # alt ft -> meters

            self.aircraft_state.tas_meters_per_sec = tas

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
                aircraft_state=self.aircraft_state,
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
        nb_vertical_points: int,
        cas_climb: float,
        mach_cruise: float,
        cas_descent: float,
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

        # COORDINATES
        lon_start, lat_start = p0.lon, p0.lat
        lon_end, lat_end = p1.lon, p1.lat

        # CLIMB, DESCENT: rocd (in ft/min)
        rocd_climb = 1500
        rocd_descent = 2000

        # CLIMB: cas (in m/s) and mach
        cas_climb1 = cas_climb * kts
        cas_climb2 = cas_climb1
        mach_climb = mach_cruise

        # DESCENT: cas (in m/s) and mach
        cas_descent1 = cas_descent * kts
        cas_descent2 = cas_descent1
        cas_descent3 = cas_descent1
        mach_descent = mach_cruise

        # CRUISE: mach
        mach_cruise = mach_cruise  # mach_cruise
        assert mach_cruise < 1, "Mach number should be less than 1"

        # ALTITUDES
        alt_init = p0.height
        alt_toc = self.cruise_height_min * ft
        alt_max = self.cruise_height_max * ft
        alt_final = p1.height

        # HEADING, BEARING and DISTANCE
        total_distance = distance(
            lat_start, lon_start, lat_end, lon_end, h=int(alt_final - alt_init)
        )
        half_distance = total_distance / 2

        # REMOVE
        n_branches = nb_lateral_points

        # initialize an empty graph
        graph = nx.DiGraph()

        # first node is the origin
        graph.add_node(
            0,
            parent_id=-1,
            branch_id=0,
            lat=lat_start,
            lon=lon_start,
            height=alt_init,
            heading=aero_bearing(lat_start, lon_start, lat_end, lon_end),
            dist_destination=total_distance,
            dist_travelled=0,
            ts=0,
            phase="climb",
            speed=cas_climb1 / kts,
            speed_type="CAS",
        )

        # define the width of the graph
        if graph_width == "small":
            alpha = 10
        elif graph_width == "medium":
            alpha = 30
        elif graph_width == "large":
            alpha = 45
        elif graph_width == "xlarge":
            alpha = 60
        else:
            raise ValueError("Graph width not defined or incorrect.")

        angles = np.linspace(
            start=-alpha, stop=alpha, num=n_branches - 1, endpoint=True, dtype=float
        )
        angles = np.insert(angles, 0, 0)
        # print(angles)

        ########################################################################################################
        ######################### FLIGHT PHASES SETUP ##########################################################
        # CLIMB
        n_steps_climb = 10
        imposed_altitudes_climb = np.array(
            [alt_init / ft, 10_000, self.alt_crossover, alt_toc / ft]
        )
        possible_altitudes_climb = (
            np.linspace(alt_init, alt_toc, num=n_steps_climb, endpoint=True) / ft
        )
        possible_altitudes_climb = np.unique(
            np.sort(np.append(possible_altitudes_climb, imposed_altitudes_climb))
        )
        time_steps_climb = (
            np.diff(possible_altitudes_climb) / rocd_climb * 60
        )  # seconds

        # CRUISE
        n_steps_cruise = nb_forward_points
        n_steps_cruise_climb = nb_vertical_points
        possible_altitudes_cruise_climb = np.linspace(
            alt_toc, alt_max, num=n_steps_cruise_climb, endpoint=True
        )

        # DESCENT
        n_steps_descent = 5
        distance_start_descent = 150 * nm
        imposed_altitudes_descent_prep = np.array(
            [self.alt_crossover, 10_000, alt_final / ft]
        )

        ######################### END OF FLIGHT PHASES SETUP ###################################################
        ########################################################################################################

        ########################################################################################################
        ######################### BEGIN OF FLIGHT PHASE ########################################################

        branches_ids = {
            "climb": [],
            "cruise": [],
            "cruise_correction": [],
            "descent": [],
        }
        for branch_id in range(n_branches):
            parent_id = 0
            angle = angles[branch_id]
            distance_halfway = half_distance / math.cos(math.radians(angle))
            parent = graph.nodes[parent_id]
            parent_height = parent["height"]

            ###### CLIMB PHASE ######
            children_climb = []
            for index_climb, time_step_climb in enumerate(time_steps_climb):
                parent = graph.nodes[parent_id]
                parent_height = parent["height"]  # in m
                plane_heading_branch = (
                    aero_bearing(
                        lat1=parent["lat"],
                        lon1=parent["lon"],
                        lat2=lat_end,
                        lon2=lon_end,
                    )
                    + angle
                )
                height = possible_altitudes_climb[index_climb + 1] * ft

                # get the right speed according to the altitude
                if height / ft < 10_000:
                    speed_type = "CAS"
                    cas_climb = cas_climb1
                elif height / ft <= self.alt_crossover:
                    speed_type = "CAS"
                    cas_climb = cas_climb2
                else:
                    speed_type = "MACH"
                    cas_climb = mach2cas(mach_climb, parent_height)

                dt = time_step_climb
                dx = cas2tas(cas_climb, h=height) * dt

                # compute new position
                lat, lon = latlon(
                    parent["lat"], parent["lon"], d=dx, brg=plane_heading_branch
                )

                # add the new node
                graph.add_node(
                    graph.number_of_nodes(),
                    parent_id=parent_id,
                    branch_id=branch_id,
                    lat=lat,
                    lon=lon,
                    height=height,
                    heading=plane_heading_branch,
                    dist_destination=distance(lat, lon, lat_end, lon_end),
                    dist_travelled=parent["dist_travelled"] + dx,
                    ts=parent["ts"] + dt,
                    phase="climb",
                    speed=cas_climb / kts if speed_type == "CAS" else mach_climb,
                    speed_type=speed_type,
                )

                # add the edge
                graph.add_edge(parent_id, graph.number_of_nodes() - 1)

                parent_id = graph.number_of_nodes() - 1

                children_climb.append(parent_id)

            branches_ids["climb"].append(children_climb)

            distance_climb_to_destination = graph.nodes[
                branches_ids["climb"][branch_id][-1]
            ]["dist_destination"]
            distance_cruise = max(
                distance_halfway
                - graph.nodes[branches_ids["climb"][branch_id][-1]]["dist_travelled"],
                distance_climb_to_destination - distance_start_descent,
            )  # (distance_climb_to_destination - distance_start_descent)
            distance_step = distance_cruise / n_steps_cruise

            # PREPARING CRUISE, ALTITUDE CHANGES
            parent_id_after_climb = parent_id
            # FIRST CRUISE PHASE
            children_cruise = []
            dx_counter = 0
            for step_cruise_climb in range(n_steps_cruise_climb):
                children_cruise_climb = []
                parent = graph.nodes[parent_id_after_climb]
                parent_height = parent["height"]
                target_altitude = possible_altitudes_cruise_climb[step_cruise_climb]
                plane_heading_branch = (
                    aero_bearing(
                        lat1=parent["lat"],
                        lon1=parent["lon"],
                        lat2=lat_end,
                        lon2=lon_end,
                    )
                    + angle
                )

                # Allows for a step climb during cruise
                if parent_height != target_altitude:
                    cas_cruise = mach2cas(mach_cruise, parent_height)
                    dz_cruise_climb = (target_altitude - parent_height) / ft
                    dt_cruise_climb = dz_cruise_climb / rocd_climb * 60
                    dx_cruise_climb = cas_cruise * dt_cruise_climb

                    # compute new position
                    lat, lon = latlon(
                        parent["lat"],
                        parent["lon"],
                        d=dx_cruise_climb,
                        brg=plane_heading_branch,
                    )

                    # add the new node
                    graph.add_node(
                        graph.number_of_nodes(),
                        parent_id=parent_id,
                        branch_id=branch_id,
                        lat=lat,
                        lon=lon,
                        height=target_altitude,
                        heading=plane_heading_branch,
                        dist_destination=distance(lat, lon, lat_end, lon_end),
                        dist_travelled=parent["dist_travelled"] + dx_cruise_climb,
                        ts=parent["ts"] + dt_cruise_climb,
                        phase="cruise",
                        speed=mach_cruise,
                        speed_type="MACH",
                    )

                    parent_id = graph.number_of_nodes() - 1
                    children_cruise_climb.append(parent_id)

                for _ in range(n_steps_cruise):
                    parent = graph.nodes[parent_id]
                    parent_distance_travelled = parent["dist_travelled"]

                    if parent_distance_travelled > distance_halfway:
                        plane_heading_branch = aero_bearing(
                            parent["lat"], parent["lon"], lat_end, lon_end
                        )
                    else:
                        plane_heading_branch = (
                            aero_bearing(parent["lat"], parent["lon"], lat_end, lon_end)
                            + angle
                        )

                    dx = distance_step
                    dt = dx / cas2tas(mach2cas(mach_cruise, height), h=height)
                    dx_counter += dx
                    # compute new position
                    lat, lon = latlon(
                        parent["lat"], parent["lon"], d=dx, brg=plane_heading_branch
                    )

                    dist_destination = distance(lat, lon, lat_end, lon_end)

                    # add the new node
                    graph.add_node(
                        graph.number_of_nodes(),
                        parent_id=parent_id,
                        branch_id=branch_id,
                        lat=lat,
                        lon=lon,
                        height=target_altitude,
                        heading=plane_heading_branch,
                        dist_destination=dist_destination,
                        dist_travelled=parent["dist_travelled"] + dx,
                        ts=parent["ts"] + dt,
                        phase="cruise",
                        speed=mach_cruise,
                        speed_type="MACH",
                    )

                    graph.add_edge(parent_id, graph.number_of_nodes() - 1)

                    parent_id = graph.number_of_nodes() - 1

                    children_cruise_climb.append(parent_id)
                # print(f"After Cruise 1, at: {dist_destination/nm} from destination; Traveled for {dx_counter/nm} nm")
                children_cruise.append(children_cruise_climb)
            branches_ids["cruise"].append(children_cruise)

            # SECOND CRUISE PHASE
            children_cruise_correction = []
            for parent_group in branches_ids["cruise"][branch_id]:
                if parent_group == []:
                    parent_group = branches_ids["climb"][branch_id]
                children_cruise_climb_correction = []
                parent_id_after_first_cruise = parent_group[-1]

                distance_after_cruise = graph.nodes[parent_id_after_first_cruise][
                    "dist_destination"
                ]
                imposed_descent_prep = np.unique(
                    np.sort(
                        np.concatenate(
                            (
                                [
                                    graph.nodes[parent_id_after_first_cruise]["height"]
                                    / ft
                                ],
                                imposed_altitudes_descent_prep,
                            )
                        )
                    )
                )
                imposed_altitude_descent_diff = np.diff(imposed_descent_prep)
                cas_descent_profile = mach2cas(
                    mach_descent, graph.nodes[parent_id_after_first_cruise]["height"]
                )  # [mach2cas(mach_descent, graph.nodes[parent_id_after_first_cruise]["height"]), cas_descent3, cas_descent2, cas_descent1]
                imposed_times_step_descent = (
                    imposed_altitude_descent_diff / rocd_descent * 60
                )

                # compute horizontal distance
                dx_total = 0
                for _, time_step_descent in enumerate(imposed_times_step_descent):
                    dx = cas2tas(cas_descent_profile, h=height) * time_step_descent
                    dx_total += dx

                delta_distance_cruise = distance_after_cruise - dx_total

                if delta_distance_cruise < 0:
                    raise ValueError(
                        "With the current ROCD and DESCENT speed profile, the plane cannot reach the destination altitude."
                    )

                distance_step = delta_distance_cruise / 5

                parent_height = graph.nodes[parent_id_after_first_cruise]["height"]
                parent_id = parent_id_after_first_cruise
                dx_counter = 0
                for _ in range(5):
                    parent = graph.nodes[parent_id]
                    parent_height = parent["height"]
                    parent_distance_travelled = parent["dist_travelled"]

                    plane_heading_branch = aero_bearing(
                        parent["lat"], parent["lon"], lat_end, lon_end
                    )

                    dx = distance_step
                    dt = dx / cas2tas(mach2cas(mach_cruise, height), h=height)
                    dx_counter += dx

                    # compute new position
                    lat, lon = latlon(
                        parent["lat"], parent["lon"], d=dx, brg=plane_heading_branch
                    )

                    dist_destination = distance(lat, lon, lat_end, lon_end)

                    # add the new node
                    graph.add_node(
                        graph.number_of_nodes(),
                        parent_id=parent_id,
                        branch_id=branch_id,
                        lat=lat,
                        lon=lon,
                        height=parent_height,
                        heading=plane_heading_branch,
                        dist_destination=dist_destination,
                        dist_travelled=parent["dist_travelled"] + dx,
                        ts=parent["ts"] + dt,
                        phase="cruise",
                        speed=mach_cruise,
                        speed_type="MACH",
                    )

                    graph.add_edge(parent_id, graph.number_of_nodes() - 1)

                    parent_id = graph.number_of_nodes() - 1

                    children_cruise_climb_correction.append(parent_id)

                children_cruise_correction.append(children_cruise_climb_correction)
            branches_ids["cruise_correction"].append(children_cruise_correction)

            # DESCENT PHASE
            children_descent = []
            for parent_group in branches_ids["cruise_correction"][branch_id]:
                children_descent_group = []
                parent_id_after_cruise_correction = parent_group[-1]
                parent_height = graph.nodes[parent_id_after_cruise_correction]["height"]
                parent_id = parent_id_after_cruise_correction

                imposed_altitudes_descent = np.concatenate(
                    ([parent_height / ft], imposed_altitudes_descent_prep)
                )
                possible_altitudes_descent = (
                    np.linspace(
                        alt_final, parent_height, num=n_steps_descent, endpoint=True
                    )
                    / ft
                )
                possible_altitudes_descent = np.unique(
                    np.sort(
                        np.append(possible_altitudes_descent, imposed_altitudes_descent)
                    )
                )[::-1]
                time_steps_descent = (
                    -np.diff(possible_altitudes_descent)[::-1] / rocd_descent * 60
                )  # seconds

                dx_counter = 0
                for index_descent, time_step_descent in enumerate(time_steps_descent):
                    parent = graph.nodes[parent_id]
                    parent_height = parent["height"]
                    plane_heading_branch = aero_bearing(
                        parent["lat"], parent["lon"], lat_end, lon_end
                    )

                    height = possible_altitudes_descent[index_descent + 1] * ft

                    # get the right speed according to the altitude
                    if height / ft < 10_000:
                        speed_type = "CAS"
                        cas_descent = cas_descent2
                    elif height / ft <= self.alt_crossover:
                        speed_type = "CAS"
                        cas_descent = cas_descent3
                    else:
                        speed_type = "MACH"
                        cas_descent = mach2cas(mach_descent, height)

                    dt = time_step_descent
                    dx = cas2tas(cas_descent, h=height) * dt
                    dx_counter += dx

                    # compute new position
                    lat, lon = latlon(
                        parent["lat"], parent["lon"], d=dx, brg=plane_heading_branch
                    )
                    dist_destination = distance(lat, lon, lat_end, lon_end)
                    # add the new node
                    graph.add_node(
                        graph.number_of_nodes(),
                        parent_id=parent_id,
                        branch_id=branch_id,
                        lat=lat,
                        lon=lon,
                        height=height,
                        heading=plane_heading_branch,
                        dist_destination=dist_destination,
                        dist_travelled=parent["dist_travelled"] + dx,
                        ts=parent["ts"] + dt,
                        phase="descent",
                        speed=cas_descent / kts
                        if speed_type == "CAS"
                        else mach_descent,
                        speed_type=speed_type,
                    )

                    # add the edge
                    graph.add_edge(parent_id, graph.number_of_nodes() - 1)

                    parent_id = graph.number_of_nodes() - 1

                    children_descent_group.append(parent_id)
                # print(f"After Descent, at: {dist_destination/nm} from destination; Traveled for {dx_counter/nm} nm")
                children_descent.append(children_descent_group)

            branches_ids["descent"].append(children_descent)
        self.branches_ids = branches_ids

        ########################################################################################################
        ######################### START OF NODE CONNECTION #####################################################

        for branch_id in range(n_branches):
            ### CLIMB PHASE ###
            # connect to branch on the left
            if branch_id > 0:
                for parent_id, child_id in zip(
                    branches_ids["climb"][branch_id][:-1],
                    branches_ids["climb"][branch_id - 1],
                ):
                    graph.add_edge(parent_id, child_id + 1)

            # connect to branch on the right
            if branch_id + 1 < n_branches:
                for parent_id, child_id in zip(
                    branches_ids["climb"][branch_id][:-1],
                    branches_ids["climb"][branch_id + 1],
                ):
                    # print(f"{parent_id} -> {child_id+1}")
                    graph.add_edge(parent_id, child_id + 1)

            parent_climb = branches_ids["climb"][branch_id][
                -1
            ]  # last climb node from the branch
            for altitude_index in range(n_steps_cruise_climb - 1):
                child_cruise = branches_ids["cruise"][branch_id][altitude_index + 1][
                    0
                ]  # first cruise node from the branch at altitude above
                graph.add_edge(parent_climb, child_cruise)

            ### CRUISE + CORRECTION PHASES ###
            for altitude_index in range(n_steps_cruise_climb):
                current_altitude_nodes = branches_ids["cruise"][branch_id][
                    altitude_index
                ]

                ### CRUISE PHASE ###
                # connect to altitude on bottom
                if altitude_index > 0:
                    bottom_altitude_nodes = branches_ids["cruise"][branch_id][
                        altitude_index - 1
                    ]
                    for parent_id, child_id in zip(
                        current_altitude_nodes, bottom_altitude_nodes
                    ):
                        if altitude_index - 1 == 0:
                            # print(f"{parent_id} -> {child_id}")
                            graph.add_edge(parent_id, child_id)
                        else:
                            # print(f"{parent_id} -> {child_id+1}")
                            graph.add_edge(parent_id, child_id + 1)

                    # connect to branch on the left
                    if branch_id > 0:
                        for parent_id, child_id in zip(
                            current_altitude_nodes, bottom_altitude_nodes
                        ):
                            # print(f"{parent_id} -> {child_id+1}")
                            graph.add_edge(parent_id, child_id + 1)

                    # connect to branch on the right
                    if branch_id + 1 < n_branches:
                        for parent_id, child_id in zip(
                            current_altitude_nodes, bottom_altitude_nodes
                        ):
                            # print(f"{parent_id} -> {child_id+1}")
                            graph.add_edge(parent_id, child_id + 1)

                # connect to altitude on top
                if altitude_index + 1 < n_steps_cruise_climb:
                    top_altitude_nodes = (
                        branches_ids["cruise"][branch_id][altitude_index + 1]
                        if altitude_index + 1 < n_steps_cruise_climb
                        else []
                    )
                    for parent_id, child_id in zip(
                        current_altitude_nodes, top_altitude_nodes
                    ):
                        if child_id + 1 in top_altitude_nodes:
                            # print(f"{parent_id} -> {child_id+1}")
                            graph.add_edge(parent_id, child_id + 1)

                    # connect to branch on the left
                    if branch_id > 0:
                        for parent_id, child_id in zip(
                            current_altitude_nodes, top_altitude_nodes
                        ):
                            # print(f"{parent_id} -> {child_id+1}")
                            graph.add_edge(parent_id, child_id + 1)

                    # connect to branch on the right
                    if branch_id + 1 < n_branches:
                        for parent_id, child_id in zip(
                            current_altitude_nodes, top_altitude_nodes
                        ):
                            # print(f"{parent_id} -> {child_id+1}")
                            graph.add_edge(parent_id, child_id + 1)

                # connect to branch on the left
                if branch_id > 0:
                    for parent_id, child_id in zip(
                        branches_ids["cruise"][branch_id][altitude_index],
                        branches_ids["cruise"][branch_id - 1][altitude_index],
                    ):
                        # print(f"{parent_id} -> {child_id+1}")
                        graph.add_edge(parent_id, child_id + 1)

                # connect to branch on the right
                if branch_id + 1 < n_branches:
                    for parent_id, child_id in zip(
                        branches_ids["cruise"][branch_id][altitude_index],
                        branches_ids["cruise"][branch_id + 1][altitude_index],
                    ):
                        # print(f"{parent_id} -> {child_id+1}")
                        graph.add_edge(parent_id, child_id + 1)

                ## CRUISE CORRECTION PHASE ###
                # connect to altitude on bottom
                if altitude_index > 0:
                    for parent_id, child_id in zip(
                        branches_ids["cruise_correction"][branch_id][altitude_index],
                        branches_ids["cruise_correction"][branch_id][
                            altitude_index - 1
                        ][:-1],
                    ):
                        # print(f"{parent_id} -> {child_id+1}")
                        graph.add_edge(parent_id, child_id + 1)

                # connect to altitude on top
                if altitude_index + 1 < n_steps_cruise_climb:
                    for parent_id, child_id in zip(
                        branches_ids["cruise_correction"][branch_id][altitude_index],
                        branches_ids["cruise_correction"][branch_id][
                            altitude_index + 1
                        ][:-1],
                    ):
                        # print(f"{parent_id} -> {child_id+1}")
                        graph.add_edge(parent_id, child_id + 1)

                # connect to branch on the left
                if branch_id > 0:
                    for parent_id, child_id in zip(
                        branches_ids["cruise_correction"][branch_id][altitude_index],
                        branches_ids["cruise_correction"][branch_id - 1][
                            altitude_index
                        ],
                    ):
                        # print(f"{parent_id} -> {child_id+1}")
                        graph.add_edge(parent_id, child_id + 1)

                # connect to branch on the right
                if branch_id + 1 < n_branches:
                    for parent_id, child_id in zip(
                        branches_ids["cruise_correction"][branch_id][altitude_index],
                        branches_ids["cruise_correction"][branch_id + 1][
                            altitude_index
                        ],
                    ):
                        # print(f"{parent_id} -> {child_id+1}")
                        graph.add_edge(parent_id, child_id + 1)

                ### DESCENT PHASE ###
                # connect to to branch on the left
                if branch_id > 0:
                    for parent_id, child_id in zip(
                        branches_ids["descent"][branch_id][altitude_index],
                        branches_ids["descent"][branch_id - 1][altitude_index][:-1],
                    ):
                        # print(f"{parent_id} -> {child_id+1}")
                        graph.add_edge(parent_id, child_id + 1)

                # connect to branch on the right
                if branch_id + 1 < n_branches:
                    for parent_id, child_id in zip(
                        branches_ids["descent"][branch_id][altitude_index],
                        branches_ids["descent"][branch_id + 1][altitude_index][:-1],
                    ):
                        # print(f"{parent_id} -> {child_id+1}")
                        graph.add_edge(parent_id, child_id + 1)

        ######################### END OF NODE CONNECTION #######################################################
        ########################################################################################################

        return graph

    def get_network(self):
        return self.network

    def flying(
        self,
        from_: pd.DataFrame,
        to_: Tuple[float, float, int],
        phase: str,
        current_speed: float,
        current_speed_type: str,
    ) -> pd.DataFrame:
        """Compute the trajectory of a flying object from a given point to a given point

        # Parameters
            from_ (pd.DataFrame): the trajectory of the object so far
            to_ (Tuple[float, float]): the destination of the object

        # Returns
            pd.DataFrame: the final trajectory of the object
        """
        pos = from_.to_dict("records")[0]
        self.aircraft_state.update_settings(
            gw_kg=pos["mass"],
            zp_ft=pos["alt"],
        )

        # check speed type (dependent if above or below Xover)
        if current_speed_type == "CAS":
            self.aircraft_state.mach = cas2mach(current_speed * kts, h=pos["alt"] * ft)
            if pos["alt"] < self.alt_crossover:
                # print(f"Current speed: {current_speed}, type: {current_speed_type}, mach: {self.aircraft_state.mach}")
                self.aircraft_state.mach = min(
                    self.aircraft_state.mach, self.aircraft_state.mach_cruise
                )

        else:
            self.aircraft_state.mach = min(current_speed, self.aircraft_state.MMO)
        # if self.aircraft_state.zp_ft < self.alt_crossover:
        #     self.aircraft_state.mach = cas2mach(cas_mach * kts, h=pos["alt"] * ft)

        # self.aircraft_state.mach = min(
        #         self.aircraft_state.mach_cruise,
        #         self.aircraft_state.MMO,
        #         cas2mach(cas_mach * kts, h=pos["alt"] * ft)
        #     )

        lat_to, lon_to, alt_to = to_[0], to_[1], to_[2]
        dist_ = distance(
            pos["lat"],
            pos["lon"],
            lat_to,
            lon_to,
            h=abs(alt_to - self.aircraft_state.zp_ft) * ft,
        )

        data = []
        epsilon = 100
        dt = 600
        dist = dist_
        loop = 0

        # print(f"Current a/c state: {self.aircraft_state.gw_kg}")

        while dist > epsilon:
            # angle
            # print((alt_to - self.aircraft_state.zp_ft) * ft / dist)
            if dist == 0:
                angle = 0
            else:
                angle = atan((alt_to - self.aircraft_state.zp_ft) * ft / dist)

            if phase == "climb":
                self.aircraft_state.phase = PhaseEnum.CLIMB
                self.aircraft_state.rating_level = RatingEnum.MCL
            elif phase == "cruise":
                self.aircraft_state.phase = PhaseEnum.CRUISE
                self.aircraft_state.rating_level = RatingEnum.CR
            else:
                # phase = descent
                self.aircraft_state.phase = PhaseEnum.DESCENT
                self.aircraft_state.rating_level = RatingEnum.CR

            self.aircraft_state.gamma_air_deg = math.degrees(angle)
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
                    [pos["ts"], (pos["alt"] + alt_to) / 2, pos["lat"], pos["lon"]],
                    field="temperature",
                )

                dISA = temp - temperature((pos["alt"] + alt_to) / 2, disa=0)

            isa_atmosphere_settings = IsaAtmosphereSettings(d_isa=dISA)

            weather_state = self.atmosphere_service.retrieve_weather_state(
                atmosphere_settings=isa_atmosphere_settings,
                four_dimensions_state=self.aircraft_state,
            )

            self.aircraft_state.weather_state = weather_state

            delta = self.aircraft_state.weather_state.static_pressure_pa / 101325.0
            self.aircraft_state.cl = (2 * self.aircraft_state.gw_kg * 9.80665) / (
                delta
                * 101325.0
                * 1.4
                * self.propulsion_settings.sref
                * self.aircraft_state.mach**2
                * math.cos(angle)
            )

            wspd = sqrt(wn * wn + we * we)

            tas = mach2tas(self.aircraft_state.mach, alt_to * ft)  # alt ft -> meters

            self.aircraft_state.tas_meters_per_sec = tas

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

            self.aircraft_state.time_step = dt

            if self.aircraft_state.phase.name == "DESCENT":
                self.aircraft_state.tsp = 0.0
            else:
                self.aircraft_state.tsp = self.propulsion_service.compute_max_rating(
                    propulsion_settings=self.propulsion_settings,
                    aircraft_state=self.aircraft_state,
                )

            ff = self.propulsion_service.compute_total_fuel_flow_kg_per_sec(
                propulsion_settings=self.propulsion_settings,
                aircraft_state=self.aircraft_state,
            )

            pos["fuel"] = ff * dt
            mass_new = self.aircraft_state.gw_kg - pos["fuel"]

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
                h=abs(self.aircraft_state.zp_ft - alt_to)
                * ft,  # height difference in m
            )

            new_row = {
                "ts": (pos["ts"] + dt),
                "lat": ll[0],
                "lon": ll[1],
                "mass": mass_new,
                "cas": current_speed
                if current_speed_type == "CAS"
                else mach2cas(current_speed, alt_to * ft) / kts,
                "mach": current_speed
                if current_speed_type == "MACH"
                else min(
                    cas2mach(current_speed * kts, alt_to * ft),
                    self.aircraft_state.mach_cruise,
                ),
                "speed_type": current_speed_type,
                "fuel": pos["fuel"],
                "alt": alt_to,  # to be modified
                "phase": self.aircraft_state.phase,
            }

            if dist <= dist_:
                data.append(new_row)
                dist_ = dist
                pos = data[-1]

            else:
                print(dist, dist_)
                dt = int(dt / 10)
                print("going in the wrong part.")
                assert dt > 0

            # update aircraft state
            self.aircraft_state.update_settings(
                gw_kg=mass_new,
                zp_ft=alt_to,
            )

            loop += 1
        # print(f"End a/c state: {self.aircraft_state.mach}, {current_speed_type}, {current_speed}")

        return pd.DataFrame(data)

    def get_weather_interpolator(self) -> WeatherForecastInterpolator:
        weather_interpolator = None

        if self.weather_date:
            lat_min = min(self.lat1, self.lat2) - 1
            lat_max = max(self.lat1, self.lat2) + 1
            lon_min = min(self.lon1, self.lon2) - 2
            lon_max = max(self.lon1, self.lon2) + 2

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

            # print("step ", i_step)
            # print("policy = ", action[0], action[1])
            # print("New node id = ", observation.id)
            # print("Alt = ", observation.alt)
            # print("Cas/Mach = ", observation.trajectory.iloc[-1]["mach"])
            # print(observation)

            # final state reached?
            if self.is_terminal(observation):
                break
        if make_img:
            print("Final state reached")
            # clear_output(wait=True)
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
                if self.aircraft_state.MFC >= terminal_state_constraints["fuel"]:
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
            aircraft_state=aircraft_state,
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
