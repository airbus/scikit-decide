# Contributing to scikit-decide

We welcome all contributions to scikit-decide.

You can help by:

- fixing bugs (see [issues](https://github.com/airbus/scikit-decide/issues) with label "bug"),
- adding new domains or solvers to the hub in `skdecide/hub/`,
- improving the documentation,
- adding and improving educational notebooks in `notebooks/`.

This is not exhaustive.

The project is hosted on [https://github.com/airbus/scikit-decide](https://github.com/airbus/scikit-decide).
Contributions to the repository are made by submitting pull requests.


This guide is organized as follows:

- [Setting up your development environment](#setting-up-your-development-environment)
- [Guidelines to follow when preparing a contribution](#guidelines-to-follow-when-preparing-a-contribution)
- [Submitting pull requests](#submitting-pull-requests) to put your contribution in the main repository


## Setting up your development environment

### Prerequisite: minizinc

You need first to install [minizinc](https://www.minizinc.org/) (version greater than 2.6) and update the `PATH` environment variable
so that it can be found by Python. See [minizinc documentation](https://www.minizinc.org/doc-latest/en/installation.html) for more details.

### Installing from source in developer mode

> **Disclaimer**: The following process has only been tested on Linux/MacOS platforms.

In order to install scikit-decide from the source so that your modification to the library are taken into account, we recommmend using poetry.

####  Prerequisites for C++
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

#### Installation with pyenv + poetry

Here are the steps to follow:

- Clone the source and got to the "scikit-decide" root directory.
    ```shell
    git clone --recurse-submodules -j8 https://github.com/airbus/scikit-decide.git
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

      - Install poetry-dynamic-versioning in poetry root env
          ```shell
          poetry self add poetry-dynamic-versioning
          ```

      - Specify to poetry the python version to use so that it creates the appropriate virtual environment.
          ```shell
          poetry env use 3.8.11
          ```

      - Preinstall gym 0.21.0 with appropriate option to avoid an error during installation
        (see this [issue](https://github.com/openai/gym/issues/3176)
        and this [solution](https://github.com/python-poetry/poetry/issues/3433#issuecomment-840509576)):
          ```shell
          poetry run python -m pip install "pip==22"  # starting with pip 23.1, gym 0.21.0 is not intallable anymore
          poetry run python -m pip install "setuptools<67"  # starting with setuptools 67, gym 0.21.0 is not intallable anymore
          poetry run python -m pip uninstall -y wheel  # wheel must not be here to fall back directly to python setup.py
          poetry run python -m pip install gym==0.21.0 --no-use-pep517
          ```

      - Install all dependencies as defined in `poetry.lock`, build and install the c++ library.
          ```shell
          rm -rf build  # removing previous build
          poetry install --extras all
          ```

#### Alternate installation with conda + poetry

You can also use conda rather than pyenv. It can be useful when you cannot install poetry via the above method,
as it can also be installed by conda via the conda-forge channel.

- Clone the source and got to the "scikit-decide" root directory.
    ```shell
    git clone --recurse-submodules -j8 https://github.com/airbus/scikit-decide.git
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

- Install poetry-dynamic-versioning in poetry root env
    ```shell
    poetry self add poetry-dynamic-versioning
    ```

- Preinstall gym 0.21.0 with appropriate option to avoid an error during installation
  (see this [issue](https://github.com/openai/gym/issues/3176)
  and this [solution](https://github.com/python-poetry/poetry/issues/3433#issuecomment-840509576)):
    ```shell
    poetry run python -m pip install "pip==22"  # starting with pip 23.1, gym 0.21.0 is not intallable anymore
    poetry run python -m pip install "setuptools<67"  # starting with setuptools 67, gym 0.21.0 is not intallable anymore
    poetry run python -m pip uninstall -y wheel  # wheel must not be here to fall back directly to python setup.py
    poetry run python -m pip install gym==0.21.0 --no-use-pep517
    ```

- Install all dependencies as defined in `poetry.lock`, build and install the c++ library.
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

### Building the docs locally

The documentation is using [VuePress](https://v1.vuepress.vuejs.org) to generate an interactive static website.
Some pages are generated from code thanks to the Python script `docs/autodoc.py`.

#### Install the library in developer mode.

See [above](#installing-from-source-in-developer-mode) to install scikit-decide with poetry.

#### Install the documentation dependencies

The Python dependencies should have been installed in previous step,
but you still need to install the JavaScript ones (including VuePress).

First, get Yarn (package manager) by following [these installation steps](https://yarnpkg.com/en/docs/install).

Make sure you are in the "scikit-decide" root directory and install documentation dependencies:

```shell
yarn install
```

#### Define environment variables for binder links

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

#### Build the docs

Make sure you are in the "scikit-decide" root directory and using the virtual environment where you installed scikit-decide.
If you used poetry, that means prepending python commands with `poetry run`.
Then generate and serve locally the documentation with:

```shell
poetry run yarn docs:dev
```

NB: The above command will call `python docs/autodoc.py` hence the use of `poetry run`.


Open your web browser to access the documentation (by default on http://localhost:8080/scikit-decide/).

### Running unit tests

The unit tests are gathered in `tests/` folder and run with [pytest](https://docs.pytest.org/).
Providing you installed the library in developer mode as described [above](#installing-from-source-in-developer-mode),
pytest should have been already installed by poetry.

From the "scikit-decide" root directory, run unit tests with:

 ```shell
 poetry run pytest tests
```

### Running notebooks as tests

One can test programmatically that notebooks are not broken thanks to [nbmake](https://github.com/treebeardtech/nbmake) extension for pytest.

```shell
poetry run pytest --nbmake notebooks -v
```

## Guidelines to follow when preparing a contribution

### Coding style and code linting

To help maintaining the same coding style across the project, some code linters are used via [pre-commit](https://pre-commit.com/).

It is used by CI to run checks at each push, but can also be used locally.

Once installed, you can run it on all files with
```shell
pre-commit run --all-files
```
Beware that doing so, you are actually modifying the files.

You can also use it when committing:

- stage your changes: `git add your_files`,
- run pre-commit on the staged files: `pre-commit run`,
- check the changes made,
- accept them by adding modified files: `git add -u`,
- commit: `git commit`.

This can also be done automatically at each commit if you add pre-commit to git hooks with `pre-commit install`.
Beware that when doing so,
- the changes will be refused if pre-commit actually modifies the files,
- you can then accept the modifications with `git add -u`,
- you can still force a commit that violates pre-commit checks with `git commit -n` or `git commit --no-verify`.

If you prefer run pre-commit manually, you can remove the hooks with `pre-commit uninstall`.

### Notebooks

We try to give some introductory examples via notebooks available in the corresponding `notebooks/` directory.

The list of these notebooks is automatically inserted in the documentation with a title and a description.
These are actually extracted from the first cell. To enable that, each notebook should

- start with a markdown cell,
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

### Adding unit tests

- Whenever adding some code, think to add some tests to the `tests/` folder.
- Whenever fixing a bug, think to add a test that crashes before fixing the bug and does not afterwards.

Follow [above instructions](#running-unit-tests) to run them with pytest.

## Submitting pull requests

When you think you are ready to merge your modifications into the main repository, you will have to open a pull request (PR).
We can summarize the process as follows:

- Fork the repository on github.
- Clone your fork on your computer.
- Make your changes and push them to your fork.
- Do the necessary checks (see [below](#prior-checks)).
- Reorganize your commits (see [below](#reorganizing-commits)).
- Submit your pull request (see [github documentation](https://help.github.com/articles/creating-a-pull-request-from-a-fork/)).
- See if all CI checks passed on your PR.
- Wait for a review.
- Take the comments and required changes into account.

Note that a PR needs at least one review by a core developer to be merged.


You may want to add a reference to the main repository to fetch from it and (re)base your changes on it:
```shell
git remote add upstream https://github.com/airbus/scikit-decide
```

This [post](https://medium.com/google-developer-experts/how-to-pull-request-d75ac81449a5) points out good practices to follow to submit great pull requests and review them efficiently.


### Prior checks

Before submitting your pull request, think to
- [run the unit tests](#running-unit-tests),
- [check the documentation locally](#building-the-docs-locally) if you modified it,
- check you respect the coding styles by [running linters](#coding-style-and-code-linting).

If you do not, you will still be able to see the status of your PR as CI will do these checks for you.

### Reorganizing commits

On your way to implement your contribution, you will probably have lots of commits,
some modifying other ones from the same PR, or only modifying the code style.

At the end of your work, consider reorganizing them by
- squashing them into one or only a few logical commits,
- having a separate commit to reformat previous existing code if necessary,
- rewritting commit messages so that it explains the changes made and why, the "how" part being explained by the code itself
  (see this [post](https://chris.beams.io/posts/git-commit/) about what a commit message should and should not contain),
- rebasing on upstream repository master branch if it diverged too much by the time you finished.

You can use `git rebase -i` to do that, as explained in [git documentation](https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History).
