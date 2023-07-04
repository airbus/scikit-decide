
# Flight Planning Domain

Flight Planning Domain is an automated planning domain of Scikit-decide.

## Documentation

### Definition of the domain

The Fight Planning Domain can be quickly defined as :
- An origin, as ICAO code of an airport,
- A destination, as ICAO code of an airport,
- An aircraft, such as define in the OpenAP library.
- An A* solver, to compute the flight plan.

### Airways graph

Taking in consideration the definition of the domain, it will creates an three-dimensional airway graph of waypoints.
The graph is following the great circle, which represent the shortest pass between the origin and the destination.
The planner computes a plan by choosing waypoints in the graph, which are represented by 4-dimensionnal states.
There is 3 phases in the graph :
- The climbing phase
- The cruise phase
- The descent phase

The flight planning domain allows to choose a number of forward, lateral and vertical waypoints in the graph.
It is also possible to choose different width (tiny, small, normal, large, xlarge) which will increase or decrease the graph width.

### State representation

Here, the states are represented by 4 features :
- The position in the graph (x,y,z)
- The aircraft mass, which can also represent the fuel consumption (integer)
- The altitude (integer)
- The time (seconds)

### Wind interpolation

The flight planning domain can take in consideration the wind conditions.
That interpolation have a major impact on the results, as jet streams are high altitude wind which can increase or decrease the ground speed of the aircraft.
It also have an impact on the computation time of a flight plan, as the objective and heuristic function became more complex.

### Objective (or cost) functions

There is three possible objective :
- Fuel (Default)
- Distance
- Time

The choosen objective will represent the cost function, which represent the cost to go from a state to another. The aim of the algorithm is to minimize the cost.

### Heuristic functions

Given we are using an A* algorithm to compute the flight plan, we need to feed it with a heuristic function, which guide the algorithm.
For now, there is 5 different heuristic function (not admissible):
- fuel, which computes the required fuel to get to the goal. It takes in consideration the local wind & speed of the aircraft.
- time, which computes the required time to get to the goal. It takes in consideration the local wind & speed of the aircraft.
- distance, wich computes the distance to the goal.
- lazy_fuel, which propagates the fuel consummed so far.
- lazy_time, which propagates the time spent on the flight so far
- None : we give a 0 cost value, which will transform the A* algorithm into a Dijkstra-like algorithm.

### Optionnal features

The flight planning domain has several optionnal features :

#### Fuel loop

The fuel loop is a optimisation of the loaded fuel for the aircraft. It will run some flights to computes the loaded fuel, using distance objective & heuristic.

#### Constraints definition

The flight planning domain allows to define constraints such as :
- A time constraint, represented by a time windows
- A fuel constraint, represented by the maximum fuel for instance.

#### Slopes

You can define your own climbing & descending slope, which have to be between 10.0 and 25.0.

## Examples

You will find an example in the notebook.
