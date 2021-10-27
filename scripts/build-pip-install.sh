#!/bin/bash
set -ex

# upgrade python packages and install cmake and auditwheels packages

# This script is called by the build-skdecide_dev.sh and build-manylinux-wheels.sh
# scripts. In a workflow triggered by .github/workflow/build.yml or release.yml,
# this script should be called first by build-skdecide_dev.sh.
# The call from build-manylinux.sh should allow to manually run build-manylinux.sh
# , as specified in its comments


for PYTHON_VER in 3.7 3.8; do
    python$PYTHON_VER -m pip install --upgrade -q pip
    python$PYTHON_VER -m pip install build cmake
done
#install auditwheel for python 3.8 (used after the build in scripts/build-manylinux-wheels.sh)
python3.8 -m pip install -q auditwheel
