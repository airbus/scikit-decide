#!/bin/bash
set -ex

yum install -y git zlib-devel
yum install -y ccache

/opt/python/cp310-cp310/bin/pip install "cmake==3.31.6"
