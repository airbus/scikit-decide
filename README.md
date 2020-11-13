[![PyPI version](https://badge.fury.io/py/scikit-decide.svg)](https://badge.fury.io/py/scikit-decide) ![PyPI - License](https://img.shields.io/pypi/l/scikit-decide) ![PyPI - Downloads](https://img.shields.io/pypi/dm/scikit-decide)
# Scikit-decide for Python

Scikit-decide is an AI framework for Reinforcement Learning, Automated Planning and Scheduling.

## Installation

### 1. Make sure to have a Python 3.7+ environment

The use of a virtual environment for scikit-decide is recommended, e.g. by using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install):

    conda create --name skdecide python=3.7
    conda activate skdecide

### 2. Install the scikit-decide library

#### a. Full installation (recommended)

Make sure you are in the "scikit-decide" root directory and install with Pip:

    cd YOUR_LOCAL_PATH_TO_GIT_CLONED_SKDECIDE
    pip install .[all]

This will install the core library and additionally all dependencies required by domains/solvers in the hub (scikit-decide catalog).

Alternatively, if you wish to install only the ones required by domains (resp. solvers) from the hub, replace `[all]` in the last command by `[domains]` (resp. `[solvers]`).

#### b. Minimal installation (not recommended)

Make sure you are in the "scikit-decide" root directory and install with Pip:

    cd YOUR_LOCAL_PATH_TO_GIT_CLONED_SKDECIDE
    pip install .

This will only install the core library, which is enough if you intend to create your own domain and solver.

#### c. C++ extension installation

This extension provides several algorithms implemented in C++ that are directly available in the Python interface.
Make sure you have a recent C++ compiler with c++-17 support.
Make sure you are in the "scikit-decide" root directory:

    cd YOUR_LOCAL_PATH_TO_GIT_CLONED_SKDECIDE

Get the git submodules that are required to build the C++ extension:

    git submodule update --init --recursive

Build and install scikit-decide containing the C++ extension with Pip:

    pip install --install-option="--cpp-extension" --install-option="--cxx-compiler=<PATH_TO_YOUR_CPP_COMPILER>" --install-option="--cmake-options="<OPTIONAL_CMAKE_OPTIONS>" .\[all\] -v

CMake options are useful in case of unusual system configurations, so we recommend to try to build the C++ extension without providing cmake options.
Should you need to pass cmake options to the installer, use the same format as the standard cmake command.

if you just want to build a dstributable wheel of scikit-decide containing the compiled C++ extension, make sure you are in the "scikit-decide" root directory and build it with setuptools:

    python setup.py bdist_wheel --cpp-extension --cxx-compiler=<PATH_TO_YOUR_CPP_COMPILER> --cmake-options=<OPTIONAL_CMAKE_OPTIONS>

## Documentation

### Online

The latest documentation is [available online](https://gheprivate.intra.corp/pages/gerard-dupont/scikit-decide).

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
    pytest tests -v
