
                    _  __    _  __              __             _      __    
       _____ _____ (_)/ /__ (_)/ /_        ____/ /___   _____ (_)____/ /___ 
      / ___// ___// // //_// // __/______ / __  // _ \ / ___// // __  // _ \
     (__  )/ /__ / // ,<  / // /_ /_____// /_/ //  __// /__ / // /_/ //  __/
    /____/ \___//_//_/|_|/_/ \__/        \__,_/ \___/ \___//_/ \__,_/ \___/

<br>
<p align="center">
  <a href="https://github.com/airbus/scikit-decide/actions?query=workflow%3Abuild_test_deploy">
    <img src="https://img.shields.io/github/workflow/status/airbus/scikit-decide/build_test_deploy?logo=github&label=CI%20status" alt="actions status">
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

## Installation

### 1. Make sure to have a Python 3.7+ environment

The use of a virtual environment for scikit-decide is recommended, e.g. by using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install):

On MacOS/Linux:

    python3 -m venv /path/to/new/virtual/environment
    source /path/to/new/virtual/environment/bin/activate

On Windows:

    py -m venv \path\to\new\virtual\environment
    \path\to\new\virtual\environment\Scripts\activate

### 2. Quick install [Recommended]

    pip3 install -U pip
    pip3 install scikit-decide[all]

### 3. Install from source

    pip3 install -U pip
    curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
    git clone --recurse-submodules -j8 https://github.com/Airbus/scikit-decide.git
    cd scikit-decide
    poetry install --no-root --extras all
    poetry build 

## Documentation

### Online

The latest documentation is [available online](https://airbus.github.io/scikit-decide/).

### Locally

You can also run the documentation locally (e.g. when you are contributing to it or to access an older version).

#### 1. Install the documentation

The documentation is using [VuePress](https://v1.vuepress.vuejs.org) to generate an interactive static website.

First, get Yarn (package manager) by following [these installation steps](https://yarnpkg.com/en/docs/install).

Make sure you are in the "scikit-decide" root directory and install documentation dependencies:

    cd YOUR_LOCAL_PATH_TO_GIT_CLONED_SKDECIDE
    yarn install

#### 2. Access the documentation

Make sure you are in the "scikit-decide" root directory and start the local documentation server:

    cd YOUR_LOCAL_PATH_TO_GIT_CLONED_SKDECIDE
    yarn docs:dev

Open your web browser to access the documentation (by default on http://localhost:8080).

## Examples

The examples can be found in the `/examples` folder, showing how to import or define a domain, and how to run or solve it. Most of the examples rely on scikit-decide Hub, an extensible catalog of domains/solvers.

Some examples are automatically embedded as Python notebooks in the `Examples` section of the documentation.

### Playground

The best example to try out scikit-decide capabilities might be `examples/full_multisolve.py`. This interactive console experience makes it easy to pick a domain among a pre-defined catalog selection:

- Simple Grid World
- Maze
- Mastermind
- Cart Pole (OpenAI Gym)
- Mountain Car continuous (OpenAI Gym)
- ATARI Pacman (OpenAI Gym)

...and then solve it with any compatible solver (detected automatically) among following selection:

- Random walk
- Simple greedy
- Lazy A* (classical planning)
- PPO: Proximal Policy Optimization (deep reinforcement learning)
- POMCP: Partially Observable Monte-Carlo Planning (online planning for POMDP)
- CGP: Cartesian Genetic Programming (evolution strategy)
- IW: Iterated Width search (width-based planning)

**Note**: some requirements declared in above solvers still need fine-tuning, so in some cases an auto-detected compatible solver may still be unable to solve a domain (for now).

These combinations are particularly efficient if you want to try them out:

- Simple Grid World -> Lazy A*
- Maze -> Lazy A*
- Mastermind -> POMCP: Partially Observable Monte-Carlo Planning
- Cart Pole -> PPO: Proximal Policy Optimization
- Mountain Car continuous -> CGP: Cartesian Genetic Programming
- ATARI Pacman -> Random walk

**Warning**: some domains/solvers might require extra manual setup steps to work at 100%. In the future, each scikit-decide hub entry might have a dedicated help page to list them, but in the meantime please refer to this:

- [domain] OpenAI Gym ones -> [gym](http://gym.openai.com/docs/#installation) for loading Gym environments not included by default
- [solver] PPO: Proximal Policy Optimization -> see [Stable Baselines installation](https://stable-baselines.readthedocs.io/en/master/guide/install.html)
- [solver] IW: Iterated Width search (same for AOstar, Astar, BFWS) -> special C++ compilation (see Installation 2.c. above)

## Unit tests

Pytest is required to run unit tests (`pip install pytest`).

Make sure you are in the "scikit-decide" root directory and run unit tests (the "-v" verbose mode is optional but gives additional details):

    cd YOUR_LOCAL_PATH_TO_GIT_CLONED_SKDECIDE
    poetry run pytest -vv -s tests/solvers/cpp
