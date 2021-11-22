# Contributing to scikit-decide

This page is intended to help people wanting to contribute to the library.

For now is it mainly about how to test locally changes made to the library.
In the future, it will also cover guidelines to follow when contributing.

## Installing from source in developer mode

> **Disclaimer**: The following process has only been tested on Linux/MacOS platforms.

In order to install scikit-decide from the source so that your modification to the library are taken into account, we recommmend using poetry.

###  Prerequisites for C++
To build the  c++ part of the library,
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

## Building the docs locally

The documentation is using [VuePress](https://v1.vuepress.vuejs.org) to generate an interactive static website.
Some pages are generated from code thanks to the Python script `docs/autodoc.py`.

### Install the library in developer mode.

See [above](#installing-from-source-developer-mode) to install scikit-decide with poetry.

### Install the documentation dependencies

The Python dependencies should have been installed in previous step,
but you still need to install the JavaScript ones (including VuePress).

First, get Yarn (package manager) by following [these installation steps](https://yarnpkg.com/en/docs/install).

Make sure you are in the "scikit-decide" root directory and install documentation dependencies:

```shell
yarn install
```

### Define environment variables for binder links

In order to define appropriate links for notebooks (github source + launching on binder), we need several environment variables:
- AUTODOC_BINDER_ENV_GH_REPO_NAME: name of the github repository hosting the binder environment
- AUTODOC_BINDER_ENV_GH_BRANCH: branch hosting the binder environment
- AUTODOC_NOTEBOOKS_REPO_URL: url of the content repository for the notebooks
- AUTODOC_NOTEBOOKS_BRANCH: branch containing the notebooks

For instance:
```shell
export AUTODOC_BINDER_ENV_GH_REPO_NAME="airbus/scikit-decide"
export AUTODOC_BINDER_ENV_GH_BRANCH="binder"
current_repo_url_withdotgit=$(git remote get-url origin)
export AUTODOC_NOTEBOOKS_REPO_URL=${current_repo_url_withdotgit/.git/}
export AUTODOC_NOTEBOOKS_BRANCH=$(git branch --show-current)
```

### Build the docs

Make sure you are in the "scikit-decide" root directory and using the virtual environment where you installed scikit-decide.
If you used poetry, that means prepending python commands with `poetry run`.
Then generate and serve locally the documentation with:

```shell
poetry run yarn docs:dev
```

NB: The above command will call  `python docs/autodoc.py` hence the use of `poetry run`.


Open your web browser to access the documentation (by default on http://localhost:8080/scikit-decide/).

## Notebooks

We try to give some introductory examples via notebooks available in the corresponding `notebooks/` directory.

### Integration in the documentation

The list of these notebooks is automatically inserted in the documentation with a title and a description.
These are actually extracted from the first cell. To enable that, each notebook should

- starts with a markdown cell,
- its first line being the title starting with one number sign ("# "),
- the remaining lines being used as the description.

For instance:

```markdown
# Great notebook title

A quick description of the main features of the notebook.
Can be on several lines.

Can include a nice thumbnail.
![Notebook_thumbnail](https://airbus.github.io/scikit-decide/maze.png)
```

## Unit tests

Pytest is required to run unit tests. Providing you installed the library in developer mode as described [above](#installing-from-source-in-developer-mode), it should have been already installed by poetry.

From the "scikit-decide" root directory, run unit tests (the "-vv" verbose mode is optional but gives additional details) with:

 ```shell
 poetry run pytest -vv -s tests/solvers/cpp
```
