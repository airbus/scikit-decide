# Installation

[[toc]]


## Installing the latest release

### 1. Make sure to have a Python 3.7+ environment

The use of a virtual environment for scikit-decide is recommended, and you will need to ensure the environment use a Python version greater than 3.7.
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

### 2. Install scikit-decide library

#### Full install [Recommended]

Install scikit-decide library from PyPI with all dependencies required by domains/solvers in the hub (scikit-decide catalog).
```shell
pip install -U pip
pip install -U scikit-decide[all]
```

#### Minimal install
Alternatively you can choose to only install the core library, which is enough if you intend to create your own domain and solver.
```shell
pip install -U pip
pip install -U scikit-decide
```

## Installing from source [Developer mode]

> **Disclaimer**: The following process has only been tested on Linux/MacOS platforms.

### Prerequisites
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


### Installation with pyenv + poetry

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

### Alternate installation with conda + poetry

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


### Use of developer mode installation

Now you are able to use the library in developer mode (i.e. with code modifications directly taken into account)
by prefixing all commands with `poetry run`.
For instance:

- to see the list of installed packages: `poetry run pip list`  (NB: you can also use `poetry show`)
- to run the tutorial script from examples: `poetry run python examples/tutorial.py`

## Installing the documentation

You can also install the documentation locally (e.g. when you are contributing to it or to access an older version).

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
