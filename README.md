
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


- [Scikit-decide for Python](#scikit-decide-for-python)
  - [Installation](#installation)
    - [Installing the latest release](#installing-the-latest-release)
      - [1. Make sure to have a Python 3.7+ environment](#1-make-sure-to-have-a-python-37-environment)
        - [With conda (all platforms)](#with-conda-all-platforms)
        - [With pyenv + venv (Linux/MacOS)](#with-pyenv--venv-linuxmacos)
        - [With pyenv-win + venv (Windows)](#with-pyenv-win--venv-windows)
      - [2. Install scikit-decide library](#2-install-scikit-decide-library)
        - [Full install [Recommended]](#full-install-recommended)
  - [Documentation](#documentation)
  - [Examples](#examples)


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

Install scikit-decide library from PyPI with all dependencies required by domains/solvers in the hub (scikit-decide catalog) ias as easy as.
```shell
pip install -U scikit-decide[all]
```

A complete installation guide is [available online](https://airbus.github.io/scikit-decide/installation/#installing-the-latest-release)

## Documentation

The latest documentation is [available online](https://airbus.github.io/scikit-decide/).

## Examples

Finally, thanks to [binder](https://mybinder.org/) some educational interactive notebooks are provided in the [Examples section](https://airbus.github.io/scikit-decide/examples/).

More examples can be found in the `/examples` folder, showing how to import or define a domain, and how to run or solve it. Most of the examples rely on scikit-decide Hub, an extensible catalog of domains/solvers.
