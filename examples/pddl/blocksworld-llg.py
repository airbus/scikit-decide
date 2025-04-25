"""Blocksworld with 3 objects with graph-objects representation + sb3 autoregressive solver.

Blocksworld: https://en.wikipedia.org/wiki/Blocks_world
Lifted Learning Graph representation: see
    Chen, D. Z., Thiébaux, S., & Trevizan, F. (2024).
    Learning Domain-Independent Heuristics for Grounded and Lifted Planning.
    Proceedings of the AAAI Conference on Artificial Intelligence, 38(18), 20078-20086.
    https://doi.org/10.1609/aaai.v38i18.29986
sb3-autoregressive: components of the action are predicted one by one,
    taking into account the choices of the previous components.
    components that are nodes of the observation graph are deduced by a GNN
    (and value is also predicted thanks to a GNN used for feature extraction)

"""


import os

from skdecide import rollout
from skdecide.hub.domain.plado import ActionEncoding, PladoPddlDomain, StateEncoding
from skdecide.hub.solver.stable_baselines import StableBaseline
from skdecide.hub.solver.stable_baselines.autoregressive.ppo.autoregressive_ppo import (
    AutoregressiveGraphPPO,
)

pddl_examples_dir = os.path.dirname(os.path.abspath(__file__))
pddl_domains_def_dir = os.path.abspath(
    f"{pddl_examples_dir}/../../tests/domains/python/pddl_domains"
)
domain_problem_dirpath = f"{pddl_domains_def_dir}/blocks"
domain_path = f"{domain_problem_dirpath}/domain.pddl"
problem_path = f"{domain_problem_dirpath}/probBLOCKS-3-0.pddl"

domain_factory = lambda: PladoPddlDomain(
    domain_path=domain_path,
    problem_path=problem_path,
    state_encoding=StateEncoding.GYM_GRAPH_LLG,
    action_encoding=ActionEncoding.GYM_MULTIDISCRETE,
)


# Action components via GNN -> node (actions or objects) of the llg graph
domain = domain_factory()
action_components_node_flag_indices = domain.get_action_components_node_flag_indices()
with StableBaseline(
    domain_factory=domain_factory,
    algo_class=AutoregressiveGraphPPO,
    baselines_policy="HeteroGraph2NodePolicy",
    policy_kwargs=dict(
        action_components_node_flag_indices=action_components_node_flag_indices
    ),
    autoregressive_action=True,
    learn_config={"total_timesteps": 10_000},
    verbose=1,
) as solver:
    solver.solve()
    max_steps = 50
    episodes = rollout(
        domain=domain,
        solver=solver,
        max_steps=max_steps,
        num_episodes=1,
        render=False,
        return_episodes=True,
    )
