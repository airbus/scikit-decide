#!/bin/bash
set -e -x

# This is to be run by Docker inside a Docker image.
# You can test it locally on a Linux machine by installing Docker and running from this
# repo's root:
# $ docker run -e PLAT=manylinux1_x86_64 -v `pwd`:/io quay.io/pypa/manylinux1_x86_64 /io/scripts/build-manylinux-wheels.sh

# The -e just defines an environment variable PLAT=[docker name] inside the Docker:
# auditwheel can't detect the Docker name automatically.

# The -v gives a directory alias for passing files in and out of the Docker.
# (/io is arbitrary). E.g the setup.py script can be accessed in the Docker via
# /io/setup.py quay.io/pypa/manylinux1_x86_64 is the full Docker image name. Docker
# downloads it automatically.

# The last argument is a shell command that the Docker will execute. Filenames must be
# from the Docker's perspective.

# Wheels are initially generated as you would usually, but put in a temp directory temp-wheels.
# The pip-cache is optional but can speed up local builds having a real permanent pip-cache dir.
mkdir -p /io/pip-cache
mkdir -p /io/temp-wheels

# Clean out any old existing wheels.
find /io/temp-wheels/ -type f -delete


#ccache setup
if test -d .ccache; then
    export CMAKE_CXX_COMPILER_LAUNCHER=ccache
    export CMAKE_C_COMPILER_LAUNCHER=ccache
    ccache --set-config cache_dir=$PWD/.ccache
    ccache --set-config max_size=80M
    ccache --set-config compression=true
fi

# pip upgrade and install cmake and auditwheel if it isn't already
$(dirname $(realpath "$0"))/build-pip-install.sh

# Uses the Python version requested as first argument, otherwise use 3.8 by default
PYTHON_VERSION=${1:-3.8}
#
for PYBIN in /opt/python/cp${PYTHON_VERSION/./}*/bin; do
    (cd /io/ && "${PYBIN}/python" -m build --sdist --wheel --outdir /io/temp-wheels)
    (cd /io/ && rm -rf build)
done

echo $PLAT

# Wheels aren't considered manylinux unless they have been through
# auditwheel. Audited wheels go in /io/dist/.
mkdir -p /io/dist/

for whl in /io/temp-wheels/*.whl; do
    auditwheel repair "$whl" --plat $PLAT -w /io/dist/
done
