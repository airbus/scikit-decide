# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import fnmatch
import os
import platform
import shutil
import sys
from ctypes.util import find_library
from pathlib import Path

pl = []

for p in sys.path:
    for dirpath, dirs, files in os.walk(p):
        for filename in fnmatch.filter(files, "__skdecide_hub_cpp*"):
            pl.append(dirpath)
            pl.append(os.path.join(dirpath, "bin"))
            os.environ["MZN_SOLVER_PATH"] = dirpath


os.environ["PATH"] += os.pathsep + os.pathsep.join(pl)
