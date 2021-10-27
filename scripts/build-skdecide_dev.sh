#!/bin/bash
set -ex

yum install -y git zlib-devel

# pip upgrade and install cmake and auditwheel packages
$(dirname $(realpath "$0"))/build-pip-install.sh
