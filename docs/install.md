# Installation

## Prerequisites

### Python 3.9+ environment

The use of a virtual environment for scikit-decide is recommended, and you will need to ensure that the environment use a Python version greater than 3.9.
This can be achieved either by using [conda](https://docs.conda.io/en/latest/) or by using [pyenv](https://github.com/pyenv/pyenv) (or [pyenv-win](https://github.com/pyenv-win/pyenv-win) on windows)
and [venv](https://docs.python.org/fr/3/library/venv.html) module.

The following examples show how to create a virtual environment with Python version 3.9.18 with the mentioned methods.

#### With conda (all platforms)

```shell
conda create -n skdecide python=3.9.18
conda activate skdecide
```

#### With pyenv + venv (Linux/MacOS)

```shell
pyenv install 3.9.18
pyenv shell 3.9.18
python -m venv skdecide-venv
source skdecide-venv/bin/activate
```

#### With pyenv-win + venv (Windows)

```shell
pyenv install 3.9.18
pyenv shell 3.9.18
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


## Known issues

### Pygrib

When installing [pygrib](https://jswhit.github.io/pygrib/index.html) on MacOS ARM (in dependencies of `scikit-decide[all]`),
no wheel exists on PyPI and there is issues when pip tries to build it.
You can overcome this by first installing `eccodes` which provides GRIB header files required to build the `pygrib` wheel:
```shell
brew install eccodes
```
Then, reinstall `scikit-decide[all]` with pip.

If the issue persists, you can try to install the pygrib package available on conda-forge:
```shell
conda install -c conda-forge pygrib
```

### Minizinc

If you plan to use the solver for scheduling domains `DOSolver`
that wraps [discrete-optimization](https://github.com/airbus/discrete-optimization) solvers,
keep in mind that some of them are based on [minizinc](https://www.minizinc.org/).
In that case, you may need to install minizinc binary (version greater than 2.8) and update the `PATH` environment variable
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
curl -o minizinc_install/minizinc -L https://github.com/MiniZinc/MiniZincIDE/releases/download/2.8.5/MiniZincIDE-2.8.5-x86_64.AppImage
chmod +x minizinc_install/minizinc
export PATH="$(pwd)/minizinc_install/":$PATH
minizinc --version
```
Else, this is still possible by extracting the files:
```
mkdir minizinc_install
cd minizinc_install
curl -o minizinc.AppImage -L https://github.com/MiniZinc/MiniZincIDE/releases/download/2.8.5/MiniZincIDE-2.8.5-x86_64.AppImage
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
curl -o minizinc.dmg -L https://github.com/MiniZinc/MiniZincIDE/releases/download/2.8.5/MiniZincIDE-2.8.5-bundled.dmg
hdiutil attach minizinc.dmg
cp -R /Volumes/MiniZinc*/MiniZincIDE.app minizinc_install/.
export PATH="$(pwd)/minizinc_install/MiniZincIDE.app/Contents/Resources":$PATH
minizinc --version
```

#### Windows command line
Works on Windows Server 2022 with bash shell:
```
mkdir minizinc_install
curl -o minizinc_setup.exe -L https://github.com/MiniZinc/MiniZincIDE/releases/download/2.8.5/MiniZincIDE-2.8.5-bundled-setup-win64.exe
cmd //c "minizinc_setup.exe /verysilent /currentuser /norestart /suppressmsgboxes /sp"
export PATH="~/AppData/Local/Programs/MiniZinc":$PATH
minizinc --version
```
