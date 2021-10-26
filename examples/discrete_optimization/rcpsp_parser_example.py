# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Utilities functions to parse RCPSP data files used in other examples scripts.

from __future__ import annotations

import os

path_to_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/rcpsp/")
files_available = [os.path.join(path_to_data, f) for f in os.listdir(path_to_data)]


def get_data_available():
    files = [f for f in os.listdir(path_to_data) if "pk" not in f and "json" not in f]
    return [os.path.join(path_to_data, f) for f in files]


def get_complete_path(root_path: str) -> str:  # example root_path="j101.sm"
    l = [f for f in get_data_available() if root_path in f]
    if len(l) > 0:
        return l[0]
    return None
