# AIRLAPS for Python

AIRLAPS is an AI framework for Reinforcement Learning, Automated Planning and Scheduling.

Open sourcing of the library is due in September 2019 on [Github](https://github.com).

## Installation

### 1. Make sure to have a Python 3.7+ environment

The use of a virtual environment for AIRLAPS is recommended, e.g. by using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install):

    conda create --name airlaps python=3.7
    conda activate airlaps

### 2. Install the AIRLAPS library

Make sure you are in the "AIRLAPS for Python" root directory and install with Pip:

    cd YOUR_LOCAL_PATH_TO_GIT_CLONED_AIRLAPS/python
    pip install .

## Documentation (work in progress)

The documentation is currently being updated for more control and flexibility, using [VuePress](https://v1.vuepress.vuejs.org) to generate an interactive static website.

For now, only the new Reference part is (almost) complete - work is still needed for the Guide part (getting started, video tutorials, hosted notebooks and code templates).

The documentation will be hosted online once AIRLAPS is open sourced. Until then, follow the steps below to access the documentation locally:

### 1. Install the documentation

Get Yarn (package manager) by following [these installation steps](https://yarnpkg.com/en/docs/install).

Make sure you are in the "AIRLAPS for Python" root directory and install documentation dependencies:

    cd YOUR_LOCAL_PATH_TO_GIT_CLONED_AIRLAPS/python
    yarn install

### 2. Access the documentation

Make sure you are in the "AIRLAPS for Python" root directory and start the local documentation server:

    cd YOUR_LOCAL_PATH_TO_GIT_CLONED_AIRLAPS/python
    yarn docs:dev

Open your web browser to access the documentation (by default on http://localhost:8080).

## Examples

**Warning**: the examples whose filename starts with an underscore are currently being migrated to the new API and might not be working in the meantime (same goes for domains/solvers inside `airlaps/catalog` which are being migrated to AIRLAPS Hub).

The examples can be found in the `/examples` folder, showing how to import or define a domain, and how to run or solve it. Most of the examples rely on AIRLAPS Hub, an extensible catalog of domains/solvers inspired by [PyTorch Hub](https://pytorch.org/hub). When loading a domain/solver from the hub via `airlaps.hub.load(...)`, the corresponding entry will be automatically downloaded from the relevant Github repository(by default `Airbus-AI-Research/AIRLAPS` but any other repo can be specified) and cached for next time.

**Note**: Since AIRLAPS is not yet open sourced on Github, the auto-download from the hub will not work out of the box and thus requires these manual steps:

- (Optional) Customize your AIRLAPS Hub cache directory:
    - Set a `AIRLAPS_HOME` environment variable to the desired directory (on Mac, [EnvPane](https://github.com/hschmidt/EnvPane#installation) is handy to set environment variables); if not set, the default `AIRLAPS_HOME` is assumed to be `~/.cache`
    - The AIRLAPS Hub cache directory is then `AIRLAPS_HOME/hub`
- Copy `YOUR_LOCAL_PATH_TO_GIT_CLONED_AIRLAPS/python/hub` inside `AIRLAPS_HOME/hub` and rename it `Airbus-AI-Research__AIRLAPS__master` (so in the end your directory structure should be `AIRLAPS_HOME/hub/Airbus-AI-Research__AIRLAPS__master`)

**Warning**: some content currently in the hub of this private repo (especially the MasterMind domain and the POMCP/CGP solvers) will require permission from their original authors before entering the public hub when open sourced.

### Playground

The best example to try out AIRLAPS capabilities might be `examples/full_multisolve.py`. This interactive console experience makes it easy to pick a domain among a pre-defined catalog selection:

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

**Warning**: some domains/solvers require extra dependencies/setup steps to work (as indicated by the warning message(s) printed when running this example). In the future, each AIRLAPS hub entry should have a dedicated `README.md` page to list them, but in the meantime please refer to this:

- [domain] All above -> [gym](http://gym.openai.com/docs/#installation) (+ [matplotlib](https://pypi.org/project/matplotlib) for Maze)
- [solver] PPO: Proximal Policy Optimization -> see [Stable Baselines installation](https://stable-baselines.readthedocs.io/en/master/guide/install.html)
- [solver] CGP: Cartesian Genetic Programming -> [gym](http://gym.openai.com/docs/#installation)
- [solver] IW: Iterated Width search -> special C++ compilation (TBD)