# AIRLAPS for Python

## Installation

### 1. Make sure to have a Python 3.6 environment (at least 3.6.1)

The use of a virtual environment for AIRLAPS is recommended, e.g. by using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install):

    conda create --name airlaps python=3.6
    conda activate airlaps

### 2. Install the AIRLAPS library

Make sure you are in the "AIRLAPS for Python" root directory:

    cd YOUR_LOCAL_PATH_TO_GIT_CLONED_AIRLAPS/python

Then, choose between minimal or full installation below.

#### a. Minimal installation of AIRLAPS (NOT recommended):

This installation has no external dependency:

    pip install .

#### b. Full installation of AIRLAPS (recommended):

This installation will also install all wrappers' dependencies:

    pip install .[wrappers]

> **Note**: some wrapped libraries might require specific additional installation steps to be used to their full extent (e.g. the deep RL library *[Stable Baselines](https://github.com/hill-a/stable-baselines)* has some [prerequisites](https://stable-baselines.readthedocs.io/en/master/guide/install.html)) - please check their installation documentation if you intend to use them.

## Documentation

On an Airbus corporate network, the documentation is [available here](http://goto/airlaps).

Otherwise, please go to the *docs* sub-folder for instructions on how to generate/build/serve the documentation (assuming a prior FULL installation of AIRLAPS).