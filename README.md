# Scikit-decide for Python

Scikit-decide is an AI framework for Reinforcement Learning, Automated Planning and Scheduling.

Open sourcing of the library is due in 2019 on [Github](https://github.com).

## Installation

### 1. Make sure to have a Python 3.7+ environment

The use of a virtual environment for scikit-decide is recommended, e.g. by using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install):

    conda create --name skdecide python=3.7
    conda activate skdecide

### 2. Install the scikit-decide library

#### a. Full installation (recommended)

Make sure you are in the "scikit-decide for Python" root directory and install with Pip:

    cd YOUR_LOCAL_PATH_TO_GIT_CLONED_SKDECIDE
    pip install .[all]

This will install the core library and additionally all dependencies required by domains/solvers in the hub (scikit-decide catalog).

Alternatively, if you wish to install only the ones required by domains (resp. solvers) from the hub, replace `[all]` in the last command by `[domains]` (resp. `[solvers]`).

#### b. Minimal installation (not recommended)

Make sure you are in the "scikit-decide for Python" root directory and install with Pip:

    cd YOUR_LOCAL_PATH_TO_GIT_CLONED_SKDECIDE
    pip install .

This will only install the core library, which is enough if you intend to create your own domain and solver.

#### c. C++ extension installation

This extension provides several algorithms implemented in C++ that are directly available in the Python interface.
Make sure you have a recent C++ compiler with c++-17 support.
Make sure you are in the "scikit-decide for Python" root directory and install with Pip:

    cd YOUR_LOCAL_PATH_TO_GIT_CLONED_SKDECIDE
    pip install --install-option="--cpp-extension" --install-option="--cxx-compiler=<PATH_TO_YOUR_CPP_COMPILER>" --install-option="--cmake-options="<OPTIONAL_CMAKE_OPTIONS>" .\[all\] -v

CMake options are useful in case of unusual system configurations, so we recommend to try to build the C++ extension without providing cmake options.
Should you need to pass cmake options to the installer, use the same format as the standard cmake command.

if you just want to build a dstributable wheel of scikit-decide containing the compiled C++ extension, make sure you are in the "scikit-decide for Python" root directory and build it with setuptools:

    cd YOUR_LOCAL_PATH_TO_GIT_CLONED_SKDECIDE
    python setup.py bdist_wheel --cpp-extension --cxx-compiler=<PATH_TO_YOUR_CPP_COMPILER> --cmake-options=<OPTIONAL_CMAKE_OPTIONS>

## Documentation (work in progress)

The documentation is currently being updated for more control and flexibility, using [VuePress](https://v1.vuepress.vuejs.org) to generate an interactive static website.

For now, only the new Reference part is (almost) complete - work is still needed for the Guide part (getting started, video tutorials, hosted notebooks and code templates).

The documentation will be hosted online once scikit-decide is open sourced. Until then, follow the steps below to access the documentation locally:

### 1. Install the documentation

Get Yarn (package manager) by following [these installation steps](https://yarnpkg.com/en/docs/install).

Make sure you are in the "scikit-decide for Python" root directory and install documentation dependencies:

    cd YOUR_LOCAL_PATH_TO_GIT_CLONED_SKDECIDE
    yarn install

### 2. Access the documentation

Make sure you are in the "scikit-decide for Python" root directory and start the local documentation server:

    cd YOUR_LOCAL_PATH_TO_GIT_CLONED_SKDECIDE
    yarn docs:dev

Open your web browser to access the documentation (by default on http://localhost:8080).

## Examples

**Warning**: the examples whose filename starts with an underscore are currently being migrated to the new API and might not be working in the meantime (same goes for domains/solvers inside `skdecide/hub`).

The examples can be found in the `/examples` folder, showing how to import or define a domain, and how to run or solve it. Most of the examples rely on scikit-decide Hub, an extensible catalog of domains/solvers.

**Warning**: some content currently in the hub (especially the MasterMind domain and the POMCP/CGP solvers) will require permission from their original authors before entering the public hub when open sourced.

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

**Warning**: some domains/solvers might require extra manual setup steps to work at 100%. In the future, each scikit-decide hub entry should have a dedicated help page to list them, but in the meantime please refer to this:

- [domain] OpenAI Gym ones -> [gym](http://gym.openai.com/docs/#installation) for loading Gym environments not included by default
- [solver] PPO: Proximal Policy Optimization -> see [Stable Baselines installation](https://stable-baselines.readthedocs.io/en/master/guide/install.html)
- [solver] IW: Iterated Width search (same for AOstar, Astar, BFWS) -> special C++ compilation (TBD)