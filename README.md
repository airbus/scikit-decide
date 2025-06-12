
                    _  __    _  __              __             _      __
       _____ _____ (_)/ /__ (_)/ /_        ____/ /___   _____ (_)____/ /___
      / ___// ___// // //_// // __/______ / __  // _ \ / ___// // __  // _ \
     (__  )/ /__ / // ,<  / // /_ /_____// /_/ //  __// /__ / // /_/ //  __/
    /____/ \___//_//_/|_|/_/ \__/        \__,_/ \___/ \___//_/ \__,_/ \___/

<br>
<p align="center">
  <a href="https://github.com/airbus/scikit-decide/actions/workflows/ci.yml?query=branch%3Amaster">
    <img src="https://img.shields.io/github/actions/workflow/status/airbus/scikit-decide/ci.yml?branch=master&logo=github&label=CI%20status" alt="actions status">
  </a>
  <a href="https://github.com/airbus/scikit-decide/tags">
    <img src="https://img.shields.io/github/tag/airbus/scikit-decide.svg?label=current%20version" alt="version">
  </a>
  <a href="https://github.com/airbus/scikit-decide/stargazers">
    <img src="https://img.shields.io/github/stars/airbus/scikit-decide.svg" alt="stars">
  </a>
  <a href="https://github.com/airbus/scikit-decide/network">
    <img src="https://img.shields.io/github/forks/airbus/scikit-decide.svg" alt="forks">
  </a>
</p>
<br>

# Scikit-decide for Python

Scikit-decide is an AI framework for Reinforcement Learning, Automated Planning and Scheduling.

This framework was initiated at [Airbus](https://www.airbus.com) AI Research and notably received contributions through the [ANITI](https://aniti.univ-toulouse.fr/en/) and [TUPLES](https://tuples.ai/) projects, and also from [ANU](https://www.anu.edu.au/).

## Main features

- Problem solving: describe your decision-making problem once and auto-match compatible solvers.
  For instance planning/scheduling problems can be solved by RL solvers and GNNs.
- Growing catalog: enjoy a growing list of domains & solvers catalog, supported by the community.
- Open & Extensible: scikit-decide is open source and is able to wrap existing state-of-the-art domains/solvers.
- Domains available:
  - [Gym(nasium)](https://gymnasium.farama.org/) environments for reinforcement learning (RL)
  - [PDDL](https://planning.wiki/) (Planning Domain Definition Language) via [unified-planning](https://github.com/aiplan4eu/unified-planning) and [plado](https://github.com/massle/plado) libraries
    - encoding in gym(nasium) spaces compatible with RL
    - graph representations for RL (inspired by [Lifted Learning Graph](https://doi.org/10.1609/aaai.v38i18.29986)) :new:
  - [RDDL](https://users.cecs.anu.edu.au/~ssanner/IPPC_2011/RDDL.pdf) (Relational Dynamic Influence Diagram Language) using [pyrddl-gym](https://github.com/pyrddlgym-project) library.
  - Flight planning, based on [openap](https://openap.dev/) or in-house Poll-Schumann for performance model
  - Scheduling, based on rcpsp problem from [discrete-optimization](https://airbus.github.io/discrete-optimization) library
  - Toy domains like: maze, mastermind, rock-paper-scissors
- Solvers available:
  - RL solvers from ray.rllib and stable-baselines3
    - existing algos with action masking
    - adaptation of RL algos for graph observation, based on GNNs from [pytorch-geometric](https://pytorch-geometric.readthedocs.io/)
    - autoregressive models with action masking component by component for parametric actions :new:
  - planning solvers from [unified-planning](https://github.com/aiplan4eu/unified-planning) library
  - RDDL solvers jax and gurobi-based based on pyRDDLGym-jax and pyRDDLGym-gurobi from [pyrddl-gym project](https://github.com/pyrddlgym-project)
  - search solvers coded in scikit-decide library:
    - A*
    - AO*
    - Improved-LAO*
    - Learning Real-Time A*
    - Best First Width Search
    - Labeled RTDP
    - Multi-Agent RTDP
    - Iterated Width search (IW)
    - Rollout IW (RIW)
    - Partially-Observable Monte Carlo Planning (POMCP)
    - Monte Carlo Tree Search Methods (MCTS)
    - Multi-Agent Heuristic meta-solver (MAHD)
  - Cartesian Genetic Programming (CGP): evolution strategy


  - scheduling solvers from [discrete-optimization](https://airbus.github.io/discrete-optimization),
    itself wrapping [ortools](https://developers.google.com/optimization), [gurobi](https://www.gurobi.com/),
    [toulbar](https://toulbar2.github.io/toulbar2/#), [minizinc](https://www.minizinc.org/),
    [deap](https://deap.readthedocs.io/) (genetic algorithm), [didppy](https://didppy.readthedocs.io/) (dynamic programming),
    and adding local search (hill climber, simulated annealing), Large Neighborhood Search (LNS),
    genetic programming based hyper-heuristic (GPHH),

- Tuning solvers hyperparameters with optuna
  - hyperparameters definition
  - automated study


## Installation

Quick version:
```shell
pip install scikit-decide[all]
```
For more details, see the [online documentation](https://airbus.github.io/scikit-decide/install).

## Documentation

The latest documentation is available [online](https://airbus.github.io/scikit-decide).

## Examples

Some educational notebooks are available in `notebooks/` folder.
Links to launch them online with [binder](https://mybinder.org/) are provided in the
[Notebooks section](https://airbus.github.io/scikit-decide/notebooks) of the online documentation.

More examples can be found as Python scripts in the `examples/` folder, showing how to import or define a domain,
and how to run or solve it. Most of the examples rely on scikit-decide Hub, an extensible catalog of domains/solvers.

## Contributing

See more about how to contribute in the [online documentation](https://airbus.github.io/scikit-decide/contribute).
