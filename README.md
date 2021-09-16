
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

### Installing the latest release

#### 1. Make sure to have a Python 3.7+ environment
  
The use of a virtual environment for scikit-decide is recommended, and you will need to ensure the environment use a Python version greater than 3.7.
This can be achieved by using [pyenv](https://github.com/pyenv/pyenv) (or [pyenv-win](https://github.com/pyenv-win/pyenv-win)) and [venv](https://docs.python.org/fr/3/library/venv.html) module as follows:

- <a name="use-pyenv"></a>Use pyenv to install an appropriate python version (3.7+).
    
    - Install the [Python build dependencies](https://github.com/pyenv/pyenv/wiki#suggested-build-environment) as suggested on pyenv github.
    - Install pyenv using [pyenv-installer](https://github.com/pyenv/pyenv-installer):
        ```shell
        curl https://pyenv.run | bash
        exec $SHELL
        ```
    - Install the chosen python version (e.g. 3.8.11):
        ```shell
        pyenv install 3.8.11
        ```
      
- Create the virtual environment with the installed python version, and activate it.
    ```shell
    pyenv shell 3.8.11
    python -m venv skdecide-venv
    source skdecide-venv
    ```   

#### 2. Install scikit-decide library

##### Full install [Recommended]

Install scikit-decide library from PyPI with all dependencies required by domains/solvers in the hub (scikit-decide catalog).
```shell
pip install -U pip
pip install -U scikit-decide[all]
```

##### Minimal install  
Alternatively you can choose to only install the core library, which is enough if you intend to create your own domain and solver.
```shell
pip install -U pip
pip install -U scikit-decide
```
  
### Installing from source [Developer mode]

In order to install scikit-decide from the source so that your modification to the library are taken into account, we recommmend using poetry.

Here are the steps to follow:

- Use pyenv to install an appropriate python version (3.7+) as explained [above](#use-pyenv).

- Clone the source and got to the "scikit-decide" root directory.
    ```shell
    git clone --recurse-submodules -j8 https://github.com/Airbus/scikit-decide.git
    cd scikit-decide
    ```
    
- Set proper python version (e.g. 3.8.11) for the scikit-decide project.
    ```shell
    pyenv local 3.8.11
    ```
  
- Update pip installer (the one that `pyenv` makes you use).
    ```shell
    pip install -U pip
    ```

- Use poetry to install the project:

    - Install [poetry](https://python-poetry.org/docs/master/#installation).
        ```shell
        curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python -
        export PATH="$HOME/.local/bin:$PATH"  # add path to poetry
        ```
    
    - Specify to poetry the python version to use so that it creates the appropriate virtual environment.
        ```shell
        poetry env use 3.8.11
        ```
      
    - Install all dependencies as defined in `poetry.lock`.
        ```shell
        rm -rf build  # removing previous build
        poetry install --extras all
        ```
        

## Documentation

### Online

The latest documentation is [available online](https://airbus.github.io/scikit-decide/).

### Locally

You can also run the documentation locally (e.g. when you are contributing to it or to access an older version).

#### 1. Install the library in developer mode.

See [above](#installing-from-source-developer-mode) to install scikit-decide with poetry.

#### 2. Install the documentation dependencies

The documentation is using [VuePress](https://v1.vuepress.vuejs.org) to generate an interactive static website.

First, get Yarn (package manager) by following [these installation steps](https://yarnpkg.com/en/docs/install).

Make sure you are in the "scikit-decide" root directory and install documentation dependencies:

```shell
yarn install
```


#### 3. Build the docs

Make sure you are in the "scikit-decide" root directory and using the virtual environment where you installed scikit-decide. 
If you used poetry, that means prepending python commands with `poetry run`.
Then generate doc with:
    
```shell
poetry run python docs/autodoc.py
```
 
#### 4. Access the documentation

Make sure you are in the "scikit-decide" root directory and start the local documentation server:

```shell
yarn docs:dev
```

Open your web browser to access the documentation (by default on http://localhost:8080/scikit-decide/).

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

Pytest is required to run unit tests. Providing you installed the library in developer mode as described [above](#installing-from-source-developer-mode), it should have been already installed by poetry.

From the "scikit-decide" root directory, run unit tests (the "-v" verbose mode is optional but gives additional details) with:

 ```shell
 poetry run pytest -vv -s tests/solvers/cpp
 ```
