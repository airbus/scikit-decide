
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


* [Installation](#installation)
  * [Installing the latest release](#installing-the-latest-release)
     * [1. Make sure to have a Python 3.7+ environment](#1-make-sure-to-have-a-python-37-environment)
        * [On Linux/MacOS](#on-linuxmacos)
        * [On Windows](#on-windows)
     * [2. Install scikit-decide library](#2-install-scikit-decide-library)
        * [Full install [Recommended]](#full-install-recommended)
        * [Minimal install](#minimal-install)
  * [Installing from source [Developer mode]](#installing-from-source-developer-mode)
* [Documentation](#documentation)
  * [Online](#online)
  * [Locally](#locally)
     * [1. Install the library in developer mode.](#1-install-the-library-in-developer-mode)
     * [2. Install the documentation dependencies](#2-install-the-documentation-dependencies)
     * [3. Build the docs](#3-build-the-docs)
     * [4. Access the documentation](#4-access-the-documentation)
* [Examples](#examples)
  * [Playground](#playground)
* [Unit tests](#unit-tests)


## Installation

### Installing the latest release

#### 1. Make sure to have a Python 3.7+ environment
  
The use of a virtual environment for scikit-decide is recommended, and you will need to ensure the environment use a Python version greater than 3.7.
This can be achieved either by using [conda](https://docs.conda.io/en/latest/) or by using [pyenv](https://github.com/pyenv/pyenv) (or [pyenv-win](https://github.com/pyenv-win/pyenv-win) on windows) 
and [venv](https://docs.python.org/fr/3/library/venv.html) module.

The following examples show how to create a virtual environment with Python version 3.8.11 with the mentioned methods. 

##### With conda (all platforms) 
 
```shell
conda create -n skdecide python=3.8.11
conda activate skdecide
```

##### With pyenv + venv (Linux/MacOS)
 
```shell
pyenv install 3.8.11
pyenv shell 3.8.11
python -m venv skdecide-venv
source skdecide-venv
```   

##### With pyenv-win + venv (Windows)

```shell
pyenv install 3.8.11
pyenv shell 3.8.11
python -m venv skdecide-venv
skdecide-venv\Scripts\activate
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

> **Disclaimer**: The following process has only been tested on Linux/MacOS platforms.

#### Prerequisites 
In order to build the library from the source and especially the c++ part, 
you need a minimal environment with c++ compiler, cmake, and boost. 
To be able to use parallelism based on openMP, you should also install libomp.
For instance, on macOS it is done via:
```shell
xcode-select --install
brew install cmake
brew install boost
brew install libomp
```


#### Installation with pyenv + poetry

In order to install scikit-decide from the source so that your modification to the library are taken into account, we recommmend using poetry.
Here are the steps to follow:

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

#### Alternate installation with conda + poetry

You can also use conda rather than pyenv. It can be useful when you cannot install poetry via the above method,
as it can also be installed by conda via the conda-forge channel.

- Clone the source and got to the "scikit-decide" root directory.
    ```shell
    git clone --recurse-submodules -j8 https://github.com/Airbus/scikit-decide.git
    cd scikit-decide
    ```
  
- Create and activate a conda environment with the proper python version for the scikit-decide project.
    ```shell
    conda create -n test_dev_skdecide python=3.8.11
    conda activate test_dev_skdecide
    ```
- Update pip installer
    ```shell
    pip install -U pip
    ```

- Install poetry in the environment
    ```shell
    conda install -c conda-forge poetry
    ```

- Install all dependencies as defined in `poetry.lock`.
    ```shell
    rm -rf build  # removing previous build
    poetry install --extras all
    ```


#### Use of developer mode installation

Now you are able to use the library in developer mode (i.e. with code modifications directly taken into account) 
by prefixing all commands with `poetry run`. 
For instance:

- to see the list of installed packages: `poetry run pip list`  (NB: you can also use `poetry show`)
- to run the tutorial script from examples: `poetry run python examples/tutorial.py`


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

> **Warning**: some domains/solvers might require extra manual setup steps to work at 100%. 
> In the future, each scikit-decide hub entry might have a dedicated help page to list them, but in the meantime please refer to this:
> - OpenAI Gym domains: [OpenAI Gym](http://gym.openai.com/docs/#installation) for loading Gym environments not included by default (e.g. atari games).

## Unit tests

Pytest is required to run unit tests. Providing you installed the library in developer mode as described [above](#installing-from-source-developer-mode), it should have been already installed by poetry.

From the "scikit-decide" root directory, run unit tests (the "-v" verbose mode is optional but gives additional details) with:

 ```shell
 poetry run pytest -vv -s tests/solvers/cpp
 ```
