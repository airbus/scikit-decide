#!/bin/bash
set -ex

yum install -y git zlib-devel
# Install the latest boost library
curl -L https://boostorg.jfrog.io/artifactory/main/release/1.76.0/source/boost_1_76_0.tar.gz -o boost-1.76.0.tar.gz \
    && tar xfz boost-1.76.0.tar.gz \
    && rm boost-1.76.0.tar.gz \
    && cd boost_1_76_0 \
    && ./bootstrap.sh --help \
    && ./bootstrap.sh --without-icu --with-libraries=headers --prefix=/usr/local \
    && ./b2 install

# pip upgrade and install cmake and auditwheel packages
$(dirname $(realpath "$0"))/build-pip-install.sh
