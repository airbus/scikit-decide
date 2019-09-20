import gym
from ray.rllib.agents.ppo import PPOTrainer

from airlaps.hub.domain.gym import GymDomain
from airlaps.hub.solver.ray_rllib import RayRLlib
from airlaps.utils import rollout


ENV_NAME = 'CartPole-v1'

domain_factory = lambda: GymDomain(gym.make(ENV_NAME))
domain = domain_factory()

# Check domain compatibility
if RayRLlib.check_domain(domain):

    # Start solving
    solver_factory = lambda: RayRLlib(PPOTrainer, train_iterations=5)
    solver = GymDomain.solve_with(solver_factory, domain_factory)
    solver.save('TEMP_RLlib')  # Save results

    # Continue solving (just to demonstrate the capability to learn further)
    solver.solve(domain_factory)
    solver.save('TEMP_RLlib')  # Save updated results

    # Test solution
    rollout(domain, solver, num_episodes=1, max_steps=1000, max_framerate=30, outcome_formatter=None)

    # Restore (latest results) from scratch and re-run
    solver = GymDomain.solve_with(solver_factory, domain_factory, load_path='TEMP_RLlib')
    rollout(domain, solver, num_episodes=1, max_steps=1000, max_framerate=30, outcome_formatter=None)
