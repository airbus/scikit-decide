# Installation

## Prerequisites

### Minizinc 2.6+

You need to install [minizinc](https://www.minizinc.org/) (version greater than 2.6) and update the `PATH` environment variable
so that it can be found by Python. See [minizinc documentation](https://www.minizinc.org/doc-latest/en/installation.html) for more details.

### Python 3.7+ environment

The use of a virtual environment for scikit-decide is recommended, and you will need to ensure that the environment use a Python version greater than 3.7.
This can be achieved either by using [conda](https://docs.conda.io/en/latest/) or by using [pyenv](https://github.com/pyenv/pyenv) (or [pyenv-win](https://github.com/pyenv-win/pyenv-win) on windows)
and [venv](https://docs.python.org/fr/3/library/venv.html) module.

The following examples show how to create a virtual environment with Python version 3.8.11 with the mentioned methods.

#### With conda (all platforms)

```shell
conda create -n skdecide python=3.8.11
conda activate skdecide
```

#### With pyenv + venv (Linux/MacOS)

```shell
pyenv install 3.8.11
pyenv shell 3.8.11
python -m venv skdecide-venv
source skdecide-venv
```

#### With pyenv-win + venv (Windows)

```shell
pyenv install 3.8.11
pyenv shell 3.8.11
python -m venv skdecide-venv
skdecide-venv\Scripts\activate
```

## Install scikit-decide library

### Full install [Recommended]

Install scikit-decide library from PyPI with all dependencies required by domains/solvers in the hub (scikit-decide catalog).
```shell
pip install -U pip
pip install -U scikit-decide[all]
```

### Minimal install
Alternatively you can choose to only install the core library, which is enough if you intend to create your own domain and solver.
```shell
pip install -U pip
pip install -U scikit-decide
```
