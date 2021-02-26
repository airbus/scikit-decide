# Examples

##  Run a Gym environment

<el-link type="primary" icon="el-icon-bottom" :underline="false" style="margin: 10px" href="/notebooks/gym_env.ipynb">Download Notebook</el-link>
<el-link type="warning" icon="el-icon-cloudy" :underline="false" style="margin: 10px" href="https://colab.research.google.com/github/airbus/scikit-decide/blob/gh-pages/notebooks/gym_env.ipynb">Run in Google Colab</el-link>

Import modules.

``` py
import gym

from skdecide.hub.domain.gym import GymDomain
from skdecide.utils import rollout
```

Select a [Gym environment](https://gym.openai.com/envs) and run 5 episodes.

``` py
ENV_NAME = 'CartPole-v1'  # or any other installed environment ('MsPacman-v4'...)

gym_domain = GymDomain(gym.make(ENV_NAME))
rollout(gym_domain, num_episodes=5, max_steps=1000, max_framerate=30, outcome_formatter=None)
gym_domain.close()  # optional but recommended to avoid Gym errors at the end
```

##  Solve a Gym environment with Reinforcement Learning

<el-link type="primary" icon="el-icon-bottom" :underline="false" style="margin: 10px" href="/notebooks/baselines_solver.ipynb">Download Notebook</el-link>
<el-link type="warning" icon="el-icon-cloudy" :underline="false" style="margin: 10px" href="https://colab.research.google.com/github/airbus/scikit-decide/blob/gh-pages/notebooks/baselines_solver.ipynb">Run in Google Colab</el-link>

Import modules.

``` py
import gym
from stable_baselines3 import PPO

from skdecide.hub.domain.gym import GymDomain
from skdecide.hub.solver.stable_baselines import StableBaseline
from skdecide.utils import rollout
```

Select a [Gym environment](https://gym.openai.com/envs) and solve it with a [Stable Baselines](https://stable-baselines3.readthedocs.io/en/master/index.html) solver wrapped in scikit-decide.
The solution is then saved (for later reuse) and assessed in rollout.

``` py
ENV_NAME = 'CartPole-v1'

domain_factory = lambda: GymDomain(gym.make(ENV_NAME))
domain = domain_factory()
if StableBaseline.check_domain(domain):
    solver_factory = lambda: StableBaseline(PPO, 'MlpPolicy', learn_config={'total_timesteps': 30000}, verbose=1)
    with solver_factory() as solver:
        GymDomain.solve_with(solver, domain_factory)
        solver.save('TEMP_Baselines')
        rollout(domain, solver, num_episodes=1, max_steps=1000, max_framerate=30, outcome_formatter=None)
```

Restore saved solution and re-run rollout.

``` py
with solver_factory() as solver:
    GymDomain.solve_with(solver, domain_factory, load_path='TEMP_Baselines')
    rollout(domain, solver, num_episodes=1, max_steps=1000, max_framerate=30, outcome_formatter=None)
```

##  Solve a Gym environment with Cartesian Genetic Programming

<el-link type="primary" icon="el-icon-bottom" :underline="false" style="margin: 10px" href="/notebooks/cgp_solver.ipynb">Download Notebook</el-link>
<el-link type="warning" icon="el-icon-cloudy" :underline="false" style="margin: 10px" href="https://colab.research.google.com/github/airbus/scikit-decide/blob/gh-pages/notebooks/cgp_solver.ipynb">Run in Google Colab</el-link>

Import modules.

``` py
import gym

from skdecide.hub.domain.gym import GymDomain
from skdecide.hub.solver.cgp import CGP  # Cartesian Genetic Programming
from skdecide.utils import rollout
```

Select a [Gym environment](https://gym.openai.com/envs) and solve it with Cartesian Genetic Programming in scikit-decide.
The solution is then assessed in rollout.

``` py
ENV_NAME = 'MountainCarContinuous-v0'

domain_factory = lambda: GymDomain(gym.make(ENV_NAME))
domain = domain_factory()
if CGP.check_domain(domain):
    solver_factory = lambda: CGP('TEMP_CGP', n_it=25)
    with solver_factory() as solver:
        GymDomain.solve_with(solver, domain_factory)
        rollout(domain, solver, num_episodes=5, max_steps=1000, max_framerate=30, outcome_formatter=None)
```

##  Create a maze domain and solve it

<el-link type="primary" icon="el-icon-bottom" :underline="false" style="margin: 10px" href="/notebooks/tutorial.ipynb">Download Notebook</el-link>
<el-link type="warning" icon="el-icon-cloudy" :underline="false" style="margin: 10px" href="https://colab.research.google.com/github/airbus/scikit-decide/blob/gh-pages/notebooks/tutorial.ipynb">Run in Google Colab</el-link>

Import modules.

``` py
from enum import Enum
from typing import *

from skdecide import *
from skdecide.builders.domain import *
from skdecide.utils import rollout
from skdecide.hub.space.gym import ListSpace, EnumSpace
from skdecide.hub.solver.lazy_astar import LazyAstar
```

Define your state space (agent positions) & action space (agent movements).

``` py
class State(NamedTuple):
    x: int
    y: int


class Action(Enum):
    up = 0
    down = 1
    left = 2
    right = 3
```

Define your domain type from a base template (DeterministicPlanningDomain here) with optional refinements (UnrestrictedActions & Renderable here).

``` py
class D(DeterministicPlanningDomain, UnrestrictedActions, Renderable):
    T_state = State  # Type of states
    T_observation = T_state  # Type of observations
    T_event = Action  # Type of events
    T_value = float  # Type of transition values (rewards or costs)
    T_info = None  # Type of additional information in environment outcome
```

Implement the maze domain by filling all non-implemented methods and adding a constructor to define the maze & start/end positions.

``` py
class MyDomain(D):

    def __init__(self, start, end, maze_str):
        self.start = start
        self.end = end
        self.maze_str = maze_str.strip()
        self.maze = self.maze_str.splitlines()

    def _get_next_state(self, memory: D.T_state, action: D.T_event) -> D.T_state:
        # Move agent according to action (except if bumping into a wall)
        next_x, next_y = memory.x, memory.y
        if action == Action.up:
            next_x -= 1
        if action == Action.down:
            next_x += 1
        if action == Action.left:
            next_y -= 1
        if action == Action.right:
            next_y += 1
        return State(next_x, next_y) if self.maze[next_x][next_y] != '.' else memory

    def _get_transition_value(self, memory: D.T_state, action: D.T_event, next_state: Optional[D.T_state] = None) -> \
            TransitionValue[D.T_value]:
        # Set cost to 1 when moving (energy cost) and to 2 when bumping into a wall (damage cost)
        return TransitionValue(cost=1 if next_state != memory else 2)

    def _get_initial_state_(self) -> D.T_state:
        # Set the start position as initial state
        return self.start

    def _get_goals_(self) -> Space[D.T_observation]:
        # Set the end position as goal
        return ListSpace([self.end])

    def _is_terminal(self, state: D.T_state) -> bool:
        # Stop an episode only when goal reached
        return self._is_goal(state)

    def _get_action_space_(self) -> Space[D.T_event]:
        # Define action space
        return EnumSpace(Action)

    def _get_observation_space_(self) -> Space[D.T_observation]:
        # Define observation space (not mandatory here)
        pass

    def _render_from(self, memory: D.T_state, **kwargs: Any) -> Any:
        # Print the maze in console with agent represented by 'o'
        cols = len(self.maze[0]) + 1
        pos = memory.x * cols + memory.y
        render = self.maze_str[:pos] + 'o' + self.maze_str[pos+1:]
        print(render)
```

Define a maze and test a random walk inside.

``` py
# Maze example ('.' represent walls, ' ' represent free space)
maze_str = '''
.....................
.   .             . .
. . . ....... ... . .
. . .   .   . . .   .
. ..... ... . . . ...
. .   .   . .   .   .
. . . . . . . ... ...
.   .   .   . .     .
............... ... .
.             .   . .
. ......... . ..... .
.   .       .       .
. . . ... ... .......
. . .   .     .     .
. ..... . ... . ... .
. .     . . . .   . .
... ... . . . ... . .
.   .   .   .   . . .
. ... ......... . . .
.   .       .     . .
.....................
'''

# Start top-left, try to reach bottom-right of this maze
domain = MyDomain(State(1, 1), State(19, 19), maze_str)

# Random walk in the maze (may sometimes reach the goal by chance)
rollout(domain, max_steps=100, render=False)
```

Pick a solver (lazy A*) and solve the maze optimally.

``` py
# Check solver compatibility with the domain
assert LazyAstar.check_domain(domain)

# Compute solution and visualize it
with LazyAstar() as solver:
    MyDomain.solve_with(solver, lambda: MyDomain(State(1, 1), State(19, 19), maze_str))
    rollout(domain, solver, max_steps=100, max_framerate=10, verbose=False)
```

##  UCT online planning with Gym environment

<el-link type="primary" icon="el-icon-bottom" :underline="false" style="margin: 10px" href="/notebooks/uct_gym_solver.ipynb">Download Notebook</el-link>
<el-link type="warning" icon="el-icon-cloudy" :underline="false" style="margin: 10px" href="https://colab.research.google.com/github/airbus/scikit-decide/blob/gh-pages/notebooks/uct_gym_solver.ipynb">Run in Google Colab</el-link>

Import modules.

``` py
import gym
import numpy as np
from typing import Callable

from skdecide.hub.domain.gym import DeterministicGymDomain, GymDiscreteActionDomain
from skdecide.hub.solver.mcts import UCT
from skdecide.utils import rollout
```

Select a [Gym environment](https://gym.openai.com/envs) and horizon parameter.

``` py
ENV_NAME = 'CartPole-v0'
HORIZON = 200
```

Define a specific UCT domain by combining Gym domain templates.

``` py
class D(DeterministicGymDomain, GymDiscreteActionDomain):
    pass


class GymUCTDomain(D):
    """This class wraps a cost-based deterministic OpenAI Gym environment as a domain
        usable by a UCT planner

    !!! warning
        Using this class requires OpenAI Gym to be installed.
    """

    def __init__(self, gym_env: gym.Env,
                       set_state: Callable[[gym.Env, D.T_memory[D.T_state]], None] = None,
                       get_state: Callable[[gym.Env], D.T_memory[D.T_state]] = None,
                       discretization_factor: int = 3,
                       branching_factor: int = None,
                       max_depth: int = 50) -> None:
        """Initialize GymUCTDomain.

        # Parameters
        gym_env: The deterministic Gym environment (gym.env) to wrap.
        set_state: Function to call to set the state of the gym environment.
                   If None, default behavior is to deepcopy the environment when changing state
        get_state: Function to call to get the state of the gym environment.
                   If None, default behavior is to deepcopy the environment when changing state
        discretization_factor: Number of discretized action variable values per continuous action variable
        branching_factor: if not None, sample branching_factor actions from the resulting list of discretized actions
        max_depth: maximum depth of states to explore from the initial state
        """
        DeterministicGymDomain.__init__(self,
                                        gym_env=gym_env,
                                        set_state=set_state,
                                        get_state=get_state)
        GymDiscreteActionDomain.__init__(self,
                                         discretization_factor=discretization_factor,
                                         branching_factor=branching_factor)
        gym_env._max_episode_steps = max_depth
```

Solve the domain with UCT solver in "realtime".

``` py
domain_factory = lambda: GymUCTDomain(gym_env=gym.make(ENV_NAME),
                                      discretization_factor=3,
                                      max_depth=HORIZON)
domain = domain_factory()

if UCT.check_domain(domain):
    solver_factory = lambda: UCT(domain_factory=domain_factory,
                                 time_budget=200,  # 200 ms,
                                 rollout_budget=100,
                                 transition_mode=UCT.Options.TransitionMode.Sample,
                                 continuous_planning=True,
                                 parallel=False, debug_logs=False)

    with solver_factory() as solver:
        GymUCTDomain.solve_with(solver, domain_factory)
        rollout(domain, solver, num_episodes=1, max_steps=HORIZON, max_framerate=30,
                outcome_formatter=lambda o: f'{o.observation} - cost: {o.value.cost:.2f}')
```

##  IW online planning with Gym environment

<el-link type="primary" icon="el-icon-bottom" :underline="false" style="margin: 10px" href="/notebooks/iw_gym_solver.ipynb">Download Notebook</el-link>
<el-link type="warning" icon="el-icon-cloudy" :underline="false" style="margin: 10px" href="https://colab.research.google.com/github/airbus/scikit-decide/blob/gh-pages/notebooks/iw_gym_solver.ipynb">Run in Google Colab</el-link>

Import modules.

``` py
import gym
import numpy as np
from typing import Callable

from skdecide.hub.domain.gym import GymPlanningDomain, GymWidthDomain, GymDiscreteActionDomain
from skdecide.hub.solver.iw import IW
from skdecide.utils import rollout
```

Select a [Gym environment](https://gym.openai.com/envs) and horizon parameter.

``` py
ENV_NAME = 'MountainCar-v0'
HORIZON = 500
```

Define a specific IW domain by combining Gym domain templates.

``` py
class D(GymPlanningDomain, GymWidthDomain, GymDiscreteActionDomain):
    pass


class GymIWDomain(D):
    """This class wraps a cost-based deterministic OpenAI Gym environment as a domain
        usable by a width-based planner

    !!! warning
        Using this class requires OpenAI Gym to be installed.
    """

    def __init__(self, gym_env: gym.Env,
                       set_state: Callable[[gym.Env, D.T_memory[D.T_state]], None] = None,
                       get_state: Callable[[gym.Env], D.T_memory[D.T_state]] = None,
                       termination_is_goal: bool = True,
                       continuous_feature_fidelity: int = 1,
                       discretization_factor: int = 3,
                       branching_factor: int = None,
                       max_depth: int = 50) -> None:
        """Initialize GymIWDomain.

        # Parameters
        gym_env: The deterministic Gym environment (gym.env) to wrap.
        set_state: Function to call to set the state of the gym environment.
                   If None, default behavior is to deepcopy the environment when changing state
        get_state: Function to call to get the state of the gym environment.
                   If None, default behavior is to deepcopy the environment when changing state
        termination_is_goal: True if the termination condition is a goal (and not a dead-end)
        continuous_feature_fidelity: Number of integers to represent a continuous feature
                                     in the interval-based feature abstraction (higher is more precise)
        discretization_factor: Number of discretized action variable values per continuous action variable
        branching_factor: if not None, sample branching_factor actions from the resulting list of discretized actions
        max_depth: maximum depth of states to explore from the initial state
        """
        GymPlanningDomain.__init__(self,
                                   gym_env=gym_env,
                                   set_state=set_state,
                                   get_state=get_state,
                                   termination_is_goal=termination_is_goal,
                                   max_depth=max_depth)
        GymDiscreteActionDomain.__init__(self,
                                         discretization_factor=discretization_factor,
                                         branching_factor=branching_factor)
        GymWidthDomain.__init__(self, continuous_feature_fidelity=continuous_feature_fidelity)
        gym_env._max_episode_steps = max_depth
```

Solve the domain with IW solver in "realtime".

``` py
domain_factory = lambda: GymIWDomain(gym_env=gym.make(ENV_NAME),
                                     termination_is_goal=True,
                                     continuous_feature_fidelity=1,
                                     discretization_factor=3,
                                     max_depth=HORIZON)
domain = domain_factory()

if IW.check_domain(domain):
    solver_factory = lambda: IW(domain_factory=domain_factory,
                                state_features=lambda d, s: d.bee1_features(
                                                                np.append(
                                                                    s._state,
                                                                    s._context[3].value.reward if s._context[3] is not None else 0)),
                                use_state_feature_hash=False,
                                node_ordering=lambda a_gscore, a_novelty, a_depth, b_gscore, b_novelty, b_depth: a_novelty > b_novelty,
                                parallel=False, debug_logs=False)

    with solver_factory() as solver:
        GymIWDomain.solve_with(solver, domain_factory)
        rollout(domain, solver, num_episodes=1, max_steps=HORIZON, max_framerate=30,
                outcome_formatter=lambda o: f'{o.observation} - cost: {o.value.cost:.2f}')
        # value, steps = simple_rollout(domain_factory(), solver, HORIZON)
        # print('value:', value)
        # print('steps:', steps)
        print('explored:', solver.get_nb_of_explored_states())
        print('pruned:', solver.get_nb_of_pruned_states())
        filter_intermediate_scores = []
        current_score = None
        for score in solver.get_intermediate_scores():
            if current_score is None or current_score != score[2]:
                current_score = score[2]
                filter_intermediate_scores.append(score)
        print('Intermediate scores:' + str(filter_intermediate_scores))
```

