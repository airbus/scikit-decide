from skdecide import rollout
from skdecide.hub.solver.stable_baselines import StableBaseline
from skdecide.hub.solver.stable_baselines.autoregressive.common.buffers import (
    ApplicableActionsRolloutBuffer,
)
from skdecide.hub.solver.stable_baselines.autoregressive.ppo.autoregressive_ppo import (
    AutoregressivePPO,
)


def test_autoregressive_ppo_w_gym_env(graph_walk_env):
    env = graph_walk_env

    algo = AutoregressivePPO(
        "MlpPolicy",
        env,
        n_steps=100,
    )
    algo.learn(total_timesteps=500)

    # rollout
    obs, info = env.reset()
    terminal = False
    i_step = 0
    print(f"#{i_step}: obs={obs}, terminal={terminal}")

    max_steps = 20

    while i_step < max_steps and not terminal:
        i_step += 1
        action, _ = algo.predict(obs, action_masks=env.action_masks())
        obs, reward, terminal, truncated, info = env.step(action)
        print(f"#{i_step}: action={action}, obs={obs}, terminal={terminal}")

    assert i_step < max_steps  # optimal would be 2, but not always found...


def test_autoregressive_ppo_w_skdecide_domain(graph_walk_domain_factory):
    domain_factory = graph_walk_domain_factory
    with StableBaseline(
        domain_factory=domain_factory,
        algo_class=AutoregressivePPO,
        baselines_policy="MlpPolicy",
        autoregressive_action=True,
        learn_config={"total_timesteps": 300},
        n_steps=100,
    ) as solver:
        solver.solve()
        max_steps = 20
        episodes = rollout(
            domain=domain_factory(),
            solver=solver,
            max_steps=max_steps,
            num_episodes=1,
            render=False,
            return_episodes=True,
        )
        observations, actions, values = episodes[0]
        assert (
            len(actions) < max_steps - 1
        )  # optimal would be 2, but not always found...
