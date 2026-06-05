#!/bin/bash
set -ex

yum install -y git zlib-devel
yum install -y ccache

/opt/python/cp310-cp310/bin/pip install "cmake==4.0.3"
