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

# minizinc setup:
#  - add embedded minizinc and related solvers binaries to PATH (last position)
#  - find solver definitions *.msc directories and set MZN_SOLVER_PATH
dirpath = __path__[0]
mzn_solver_paths = ""
mzn_bin_path = ""


def _contains_minizinc_bin_directory(dirpath):
    return os.path.exists(f"{dirpath}/bin/minizinc") or os.path.exists(
        f"{dirpath}/bin/minizinc.exe"
    )


if _contains_minizinc_bin_directory(dirpath):
    # release install via pip
    mzn_solver_path = dirpath
    mzn_bin_path = f"{dirpath}/bin"
else:
    # dev install via poetry: find skdecide/hub in build directory, only one instance
    break_loop = False
    for p in sys.path:
        if break_loop:
            break
        for dirpath, dirs, files in os.walk(p):
            if len(
                fnmatch.filter(files, "__skdecide_hub_cpp*")
            ) > 0 and _contains_minizinc_bin_directory(dirpath):
                mzn_solver_path = dirpath
                mzn_bin_path = f"{dirpath}/bin"
                break_loop = True
                break
# set path and mzn solver path
if mzn_bin_path:
    os.environ["PATH"] += os.pathsep + mzn_bin_path
if mzn_solver_path:
    os.environ["MZN_SOLVER_PATH"] = mzn_solver_path
