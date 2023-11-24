import gymnasium as gym
import ray.rllib.utils
from ray.rllib.algorithms.ppo import PPO

from skdecide.core import autocast_all
from skdecide.domains import Domain
from skdecide.hub.domain.gym import GymDomain
from skdecide.hub.domain.rock_paper_scissors import RockPaperScissors
from skdecide.hub.solver.ray_rllib.ray_rllib import AsRLlibMultiAgentEnv, RayRLlib
from skdecide.utils import rollout


def test_as_rllib_env():
    domain = RockPaperScissors()
    env = AsRLlibMultiAgentEnv(domain)

    # check action space
    assert isinstance(env.action_space, gym.spaces.Dict)
    assert env.get_agent_ids() == set(env.action_space)
    for subspace in env.action_space.values():
        assert isinstance(subspace, gym.spaces.Space)

    # check observation space
    assert isinstance(env.observation_space, gym.spaces.Dict)
    assert env.get_agent_ids() == set(env.observation_space)
    for subspace in env.observation_space.values():
        assert isinstance(subspace, gym.spaces.Space)

    # checks from ray.rllib
    ray.rllib.utils.check_env(env)


def test_as_rllib_env_with_autocast_from_singleagent_to_multiagents():
    ENV_NAME = "CartPole-v1"

    upcast_domain = GymDomain(gym.make(ENV_NAME))
    autocast_all(upcast_domain, GymDomain, RayRLlib.T_domain)
    env = AsRLlibMultiAgentEnv(upcast_domain)

    # check action space
    assert isinstance(env.action_space, gym.spaces.Dict)
    assert env.get_agent_ids() == set(env.action_space)
    for subspace in env.action_space.values():
        assert isinstance(subspace, gym.spaces.Space)

    # check observation space
    assert isinstance(env.observation_space, gym.spaces.Dict)
    assert env.get_agent_ids() == set(env.observation_space)
    for subspace in env.observation_space.values():
        assert isinstance(subspace, gym.spaces.Space)

    # checks from ray.rllib
    ray.rllib.utils.check_env(env)


def test_ray_rllib_solver():
    # define domain
    domain_factory = lambda: RockPaperScissors()
    domain = domain_factory()

    # check compatibility
    assert RayRLlib.check_domain(domain)

    # define and solve
    solver_kwargs = dict(algo_class=PPO, train_iterations=1)
    solver = RockPaperScissors.solve_with(RayRLlib(**solver_kwargs), domain_factory)
    assert hasattr(solver, "_algo")

    # store
    tmp_save_dir = "TEMP_RLlib"
    solver.save(tmp_save_dir)

    # rollout
    rollout(
        domain,
        solver,
        action_formatter=lambda a: str({k: v.name for k, v in a.items()}),
        outcome_formatter=lambda o: f"{ {k: v.name for k, v in o.observation.items()} }"
        f" - rewards: { {k: v.reward for k, v in o.value.items()} }",
    )

    # load and rollout
    solver2 = RayRLlib(**solver_kwargs)
    solver2.load(tmp_save_dir, domain_factory)
    rollout(
        domain,
        solver2,
    )


def test_ray_rllib_solver_on_single_agent_domain():
    # define domain
    ENV_NAME = "CartPole-v1"
    domain_factory = lambda: GymDomain(gym.make(ENV_NAME))
    domain = domain_factory()

    # check compatibility
    assert RayRLlib.check_domain(domain)

    # define and solve
    solver_kwargs = dict(algo_class=PPO, train_iterations=1)
    solver = GymDomain.solve_with(RayRLlib(**solver_kwargs), domain_factory)
    assert hasattr(solver, "_algo")

    # rollout
    rollout(
        domain,
        solver,
    )
