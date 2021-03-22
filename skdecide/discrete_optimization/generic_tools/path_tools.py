# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os

def get_directory(file):
    return os.path.dirname(file)

def abspath_from_file(file, relative_path):
    return os.path.join(os.path.dirname(os.path.abspath(file)), relative_path)