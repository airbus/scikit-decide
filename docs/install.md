# Installation

## Prerequisites

### Minizinc 2.6+

You need to install [minizinc](https://www.minizinc.org/) (version greater than 2.6) and update the `PATH` environment variable
so that it can be found by Python. See [minizinc documentation](https://www.minizinc.org/doc-latest/en/installation.html) for more details.

::: tip
You can easily install minizinc from the command line, which can be useful when on cloud.
In order to make life easier to cloud users, we reproduce below the necessary lines. Please be careful that this
is not an official documentation for minizinc and that the following lines can stop working without notice
as we do not test them automatically.
:::

#### Linux command line
On a Linux distribution, you can use the bundled [minizinc AppImage](https://www.minizinc.org/doc-latest/en/installation.html#appimage).

If [FUSE](https://en.wikipedia.org/wiki/Filesystem_in_Userspace) is available:
```
mkdir minizinc_install
curl -o minizinc_install/minizinc -L https://github.com/MiniZinc/MiniZincIDE/releases/download/2.6.3/MiniZincIDE-2.6.3-x86_64.AppImage
chmod +x minizinc_install/minizinc
export PATH="$(pwd)/minizinc_install/":$PATH
minizinc --version
```
Else, this is still possible by extracting the files:
```
mkdir minizinc_install
cd minizinc_install
curl -o minizinc.AppImage -L https://github.com/MiniZinc/MiniZincIDE/releases/download/2.6.3/MiniZincIDE-2.6.3-x86_64.AppImage
chmod +x minizinc.AppImage
./minizinc.AppImage --appimage-extract
cd ..
export LD_LIBRARY_PATH="$(pwd)/minizinc_install/squashfs-root/usr/lib/":$LD_LIBRARY_PATH
export PATH="$(pwd)/minizinc_install/squashfs-root/usr/bin/":$PATH
minizinc --version
```

#### MacOs command line
```
mkdir minizinc_install
curl -o minizinc.dmg -L https://github.com/MiniZinc/MiniZincIDE/releases/download/2.6.3/MiniZincIDE-2.6.3-bundled.dmg
hdiutil attach minizinc.dmg
cp -R /Volumes/MiniZinc*/MiniZincIDE.app minizinc_install/.
export PATH="$(pwd)/minizinc_install/MiniZincIDE.app/Contents/Resources":$PATH
minizinc --version
```

#### Windows command line
Works on Windows Server 2022 with bash shell:
```
mkdir minizinc_install
curl -o minizinc_setup.exe -L https://github.com/MiniZinc/MiniZincIDE/releases/download/2.6.3/MiniZincIDE-2.6.3-bundled-setup-win64.exe
cmd //c "minizinc_setup.exe /verysilent /currentuser /norestart /suppressmsgboxes /sp"
export PATH="~/AppData/Local/Programs/MiniZinc":$PATH
minizinc --version
```

#### Skipping minizinc version check

It may happen that you do not want to install minzinc.
For instance if you need to use only a part of the library which is not relying on minizinc at all,
especially when not relying on discrete-optimization which is the actual library depending on minizinc.

This can be troublesome as the minizinc binary version is checked by [discrete-optimization](https://airbus.github.io/discrete-optimization) at library import.
However discrete-optimization provides a way to bypass this check
by setting the environment variable DO_SKIP_MZN_CHECK:
```shell
export DO_SKIP_MZN_CHECK=1
```
Please note however that the library is never tested without minizinc (or minizinc versions < 2.6).


### Python 3.7+ environment

The use of a virtual environment for scikit-decide is recommended, and you will need to ensure that the environment use a Python version greater than 3.7.
This can be achieved either by using [conda](https://docs.conda.io/en/latest/) or by using [pyenv](https://github.com/pyenv/pyenv) (or [pyenv-win](https://github.com/pyenv-win/pyenv-win) on windows)
and [venv](https://docs.python.org/fr/3/library/venv.html) module.

The following examples show how to create a virtual environment with Python version 3.8.13 with the mentioned methods.

#### With conda (all platforms)

```shell
conda create -n skdecide python=3.8.13
conda activate skdecide
```

#### With pyenv + venv (Linux/MacOS)

```shell
pyenv install 3.8.13
pyenv shell 3.8.13
python -m venv skdecide-venv
source skdecide-venv/bin/activate
```

#### With pyenv-win + venv (Windows)

```shell
pyenv install 3.8.13
pyenv shell 3.8.13
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

## Troubleshooting

You may encounter an [error when installing `gym==0.21.0`](https://github.com/openai/gym/issues/3176) which happens to be a dependency of `scikit-decide[all]`. 
This is because its installation does not respect PEP 517 which is enforced by default by last versions of pip and setuptools. 
The solution is to install it beforehand:
```shell
# preinstall gym==0.21.0 with legacy method (python setup.py) because its requirements list is broken
python -m pip install "pip==22"  # starting with pip 23.1, gym 0.21.0 is not intallable anymore
python -m pip install "setuptools<67"  # starting with setuptools 67, gym 0.21.0 is not intallable anymore
python -m pip uninstall -y wheel  # wheel must not be here to fall back directly to python setup.py
python -m pip install gym==0.21.0 --no-use-pep517
# upgrade pip
python -m pip install --upgrade pip
# install scikit-decide and remaining dependencies
pip install -U scikit-decide[all]
```

::: tip Note
Newer versions of gym or [gymnasium](https://gymnasium.farama.org/), typically greater than 0.26 are not yet possible 
because of a conflict between [`ray[rllib]`](https://github.com/ray-project/ray/issues/34396) 
and [`stable-baselines3`](https://github.com/DLR-RM/stable-baselines3/issues/1452).
:::