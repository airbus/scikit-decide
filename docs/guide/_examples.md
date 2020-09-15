# Examples

##  Run a Gym environment

<el-link type="primary" icon="el-icon-bottom" :underline="false" style="margin: 10px" href="/notebooks/gym_env.ipynb">Download Notebook</el-link>
<el-link type="warning" icon="el-icon-cloudy" :underline="false" style="margin: 10px" href="https://colab.research.google.com/github/airbus-ai-research/skdecide/gh-pages/notebooks/gym_env.ipynb">Run in Google Colab</el-link>

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
<el-link type="warning" icon="el-icon-cloudy" :underline="false" style="margin: 10px" href="https://colab.research.google.com/github/airbus-ai-research/skdecide/gh-pages/notebooks/baselines_solver.ipynb">Run in Google Colab</el-link>

Import modules.

``` py
import gym
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy

from skdecide.hub.domain.gym import GymDomain
from skdecide.hub.solver.stable_baselines import StableBaseline
from skdecide.utils import rollout
```

Select a [Gym environment](https://gym.openai.com/envs) and solve it with a [Stable Baselines](https://stable-baselines.readthedocs.io/en/master/index.html) solver wrapped in scikit-decide.
The solution is then saved (for later reuse) and assessed in rollout.

``` py
ENV_NAME = 'CartPole-v1'

domain_factory = lambda: GymDomain(gym.make(ENV_NAME))
domain = domain_factory()
if StableBaseline.check_domain(domain):
    solver_factory = lambda: StableBaseline(PPO2, MlpPolicy, learn_config={'total_timesteps': 50000}, verbose=1)
    solver = GymDomain.solve_with(solver_factory, domain_factory)
    solver.save('TEMP_Baselines')
    rollout(domain, solver, num_episodes=1, max_steps=1000, max_framerate=30, outcome_formatter=None)
```

Restore saved solution and re-run rollout.

``` py
solver = GymDomain.solve_with(solver_factory, domain_factory, load_path='TEMP_Baselines')
rollout(domain, solver, num_episodes=1, max_steps=1000, max_framerate=30, outcome_formatter=None)
```

