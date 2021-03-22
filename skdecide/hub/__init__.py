# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import fnmatch
import os
import sys

import platform
from pathlib import Path
from ctypes.util import find_library
import shutil

pl = []

for p in sys.path:
    for dirpath, dirs, files in os.walk(p):
        for filename in fnmatch.filter(files, "__skdecide_hub_cpp*"):
            pl.append(dirpath)
            pl.append(os.path.join(dirpath, 'bin'))
            os.environ["MZN_SOLVER_PATH"] = dirpath

#sys.path.extend(pl)


os.environ["PATH"] += os.pathsep + os.pathsep.join(pl)

""" path_bin = os.environ.get("PATH", "").split(os.pathsep)
print(path_bin)
path_lib = os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
path_lib.extend(os.environ.get("DYLD_LIBRARY_PATH", "").split(os.pathsep))

# Add default MiniZinc locations to the path
if platform.system() == "Darwin":
    MAC_LOCATIONS = [
        str(Path("/Applications/MiniZincIDE.app/Contents/Resources")),
        str(Path("~/Applications/MiniZincIDE.app/Contents/Resources").expanduser()),
    ]
    path_bin.extend(MAC_LOCATIONS)
    path_lib.extend(MAC_LOCATIONS)
elif platform.system() == "Windows":
    WIN_LOCATIONS = [
        str(Path("c:/Program Files/MiniZinc")),
        str(Path("c:/Program Files/MiniZinc IDE (bundled)")),
        str(Path("c:/Program Files (x86)/MiniZinc")),
        str(Path("c:/Program Files (x86)/MiniZinc IDE (bundled)")),
    ]
    path_bin.extend(WIN_LOCATIONS)
    path_lib.extend(WIN_LOCATIONS)


path_bin_list = os.pathsep.join(path_bin)
path_lib_list = os.pathsep.join(path_lib)

# Try to load the MiniZinc C API
env_backup = os.environ.copy()
os.environ["LD_LIBRARY_PATH"] = path_lib_list
os.environ["DYLD_LIBRARY_PATH"] = path_lib_list

name = "minizinc"

lib = find_library(name)

if lib and Path(lib).suffix in [".dll", ".dylib", ".so"]:
    pass
    # TODO:
    # from minizinc.API import APIDriver

    # library = cdll.LoadLibrary(lib)
    # driver = APIDriver(library)
    print("lib {} found".format(name))
else:
     # Try to locate the MiniZinc executable
    executable = shutil.which(name, path=path_bin_list)
    print("executable {} is {} in {}".format(name, executable, path_bin_list))

def goodbye(env_backup):
    os.environ.clear()
    os.environ.update(env_backup)

import atexit
atexit.register(goodbye, env_backup=env_backup) """