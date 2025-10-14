"""Blocksworld with 3 objects with graph-objects representation + sb3 autoregressive solver.

Blocksworld: https://en.wikipedia.org/wiki/Blocks_world
Graph-objects representation = Objects Binary Structure presented in
    Horčík, R., & Šír, G. (2024). Expressiveness of Graph Neural Networks in Planning Domains.
    Proceedings of the International Conference on Automated Planning and Scheduling, 34(1), 281-289.
    https://ojs.aaai.org/index.php/ICAPS/article/view/31486
sb3-autoregressive: components of the action are predicted one by one,
    taking into account the choices of the previous components.

"""

import os

from skdecide import rollout
from skdecide.hub.domain.plado import (
    ActionEncoding,
    ObservationEncoding,
    PladoTransformedObservablePddlDomain,
)
from skdecide.hub.solver.stable_baselines import StableBaseline
from skdecide.hub.solver.stable_baselines.autoregressive.ppo.autoregressive_ppo import (
    AutoregressiveGraphPPO,
)

pddl_examples_dir = os.path.dirname(os.path.abspath(__file__))
pddl_domains_def_dir = os.path.abspath(
    f"{pddl_examples_dir}/../../tests/domains/python/pddl_domains"
)

domain_path = f"{pddl_domains_def_dir}/blocks/domain.pddl"
problem_path = f"{pddl_domains_def_dir}/blocks/probBLOCKS-3-0.pddl"

domain_factory = lambda: PladoTransformedObservablePddlDomain(
    domain_path=domain_path,
    problem_path=problem_path,
    obs_encoding=ObservationEncoding.GYM_GRAPH_OBJECTS,
    action_encoding=ActionEncoding.GYM_MULTIDISCRETE,
)

# Feature extraction via GNN
with StableBaseline(
    domain_factory=domain_factory,
    algo_class=AutoregressiveGraphPPO,
    baselines_policy="GraphInputPolicy",
    autoregressive_action=True,
    learn_config={"total_timesteps": 10_000},
    verbose=1,
) as solver:
    solver.solve()
    max_steps = 50
    episodes = rollout(
        domain=domain_factory(),
        solver=solver,
        max_steps=max_steps,
        num_episodes=1,
        render=False,
        return_episodes=True,
    )


# + node prediction via GNN for actions parameters
with StableBaseline(
    domain_factory=domain_factory,
    algo_class=AutoregressiveGraphPPO,
    baselines_policy="Graph2NodePolicy",
    autoregressive_action=True,
    learn_config={"total_timesteps": 10_000},
    verbose=1,
) as solver:
    solver.solve()
    max_steps = 50
    episodes = rollout(
        domain=domain_factory(),
        solver=solver,
        max_steps=max_steps,
        num_episodes=1,
        render=False,
        return_episodes=True,
    )
