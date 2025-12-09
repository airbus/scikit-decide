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

### Installing from source in developer mode

> **Disclaimer**: The following process has only been tested on Linux/MacOS platforms.

In order to install scikit-decide from the source so that your modification to the library are taken into account, we recommend using [uv](https://docs.astral.sh/uv/).

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
You must also set some environment variables to instruct the build process on how to find boost and openMP.
For instance, on macOS, it is done via:
```shell
export Boost_ROOT=$(brew --cellar boost)/$(brew list --versions boost | tr ' ' '\n' | tail -1)
export OpenMP_ROOT=$(brew --cellar libomp)/$(brew list --versions libomp | tr ' ' '\n' | tail -1)
export CPPFLAGS="$CPPFLAGS -Xpreprocessor -fopenmp"
export CFLAGS="$CFLAGS -I$OpenMP_ROOT/include"
export CXXFLAGS="$CXXFLAGS -I$OpenMP_ROOT/include"
export LDFLAGS="$LDFLAGS -Wl,-rpath,$OpenMP_ROOT/lib -L$OpenMP_ROOT/lib -lomp"
```

#### Installation with uv

Here are the steps to follow:

- Clone the source and got to the "scikit-decide" root directory.
    ```shell
    git clone --recurse-submodules -j8 https://github.com/airbus/scikit-decide.git
    cd scikit-decide
    ```

- Set proper python version (e.g. 3.12) for the scikit-decide project.
    ```shell
    echo 3.12 > .python-version
    ```
  or add `--python=3.12` to the first uv command (`sync` or `run`).
  You can also skip this step and uv will take the current python version.

- Install uv (see [uv documentation](https://docs.astral.sh/uv/getting-started/installation/)).

- Install (and build the c++ library) the scikit-decide library in editable mode
  (it will take into account any code changes in python source), optionally with extra "all", and all dev dependencies (jupyter, pytest, ...)

  ```shell
  uv sync --extra=all -v
  ```

  ::: tip Notes
    - `--extra=all`: adds "all" extra necessary for using all solvers and domains in hub
    - `-v`: add verbosity to see what is happening during c++ library build
    - to speed-up rebuilds, the c++ build are done in a directory "build/{wheel_tag}". If you encounter build issues you can try removing the build directory.
  :::

#### Use of developer mode installation

Now you are able to use the library in developer mode (i.e. with code modifications directly taken into account)
by prefixing all commands with `uv run`.
For instance:
- to run the tutorial script from examples:
    ```shell
    uv run python examples/tutorial.py
    ```

To see the list of installed package, you can go
- either `uv tree`,
- or `uv pip list`.

See more command for uv in its [documentation](https://docs.astral.sh/uv/concepts/projects/).



::: tip
If uv or the build backend (scikit-build-core) "thinks" that your built code has changed (for instance when you switch between git branches)
it will rebuilt it automatically at next `uv run` call. It should be quite fast thanks to the use of a deterministic build directory name ("build/{wheel_tag}"),
but you can also avoid it if you know the c++ library has not changed by adding the option `--no-sync`
(be aware that it will also prevent any changes in dependencies you introduced in pyproject.toml). Ex:
```shell
uv run --no-sync python examples/tutorial.py
```
:::

### Building the docs locally

The documentation is using [VuePress](https://v1.vuepress.vuejs.org) to generate an interactive static website.
Some pages are generated from code thanks to the Python script `docs/autodoc.py`.

#### Install the documentation dependencies

The Python dependencies should have been installed in previous step,
but you still need to install the JavaScript ones (including VuePress).

First, get Yarn (package manager) by following [these installation steps](https://yarnpkg.com/en/docs/install).

Make sure you are in the "scikit-decide" root directory and install documentation dependencies:

```shell
yarn install
```

#### Define environment variables for binder and colab links  [optional]

In order to define appropriate links for notebooks (github source + launching on binder or colab), we need several environment variables:
- AUTODOC_NOTEBOOKS_REPO_URL: url of the content repository for the notebooks
- AUTODOC_NOTEBOOKS_BRANCH: branch containing the notebooks

For instance:
```shell
current_repo_url_withdotgit=$(git remote get-url origin)
export AUTODOC_NOTEBOOKS_REPO_URL=${current_repo_url_withdotgit/.git/}
export AUTODOC_NOTEBOOKS_BRANCH=$(git branch --show-current)
```

If you skip this part, the python script will try to use the above commands to fill the variables.
If `git` is missing or for some reason the commands fail, the github, binder, and colab links will simply be not generated.

#### Build the docs

Make sure you are in the "scikit-decide" root directory,
then generate and serve locally the documentation with:

```shell
uv run yarn docs:dev
```

- The above command will call `python docs/autodoc.py` hence the use of `uv run`.


Open your web browser to access the documentation (by default on [http://localhost:8080/scikit-decide/](http://localhost:8080/scikit-decide/)).

### Running unit tests

The unit tests are gathered in `tests/` folder and run with [pytest](https://docs.pytest.org/).
Providing you installed the library in developer mode as described [above](#installing-from-source-in-developer-mode),
pytest should have been already installed by uv.

From the "scikit-decide" root directory, run unit tests with:

 ```shell
 uv run pytest tests
```

### Running notebooks as tests

One can test programmatically that notebooks are not broken thanks to [nbmake](https://github.com/treebeardtech/nbmake) extension for pytest.

```shell
uv run pytest --nbmake notebooks -v
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
