#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import importlib
import logging
import pkgutil

logger = logging.getLogger(__name__)


def find_abs_modules(package, include_private=False):
    """Find names of all submodules in the package."""
    path_list = []
    for modinfo in pkgutil.walk_packages(package.__path__, f"{package.__name__}."):
        if not modinfo.ispkg:
            if include_private or not (
                modinfo.name.split(".")[-1].startswith("_")
            ):  # skip module names with leading _
                path_list.append(modinfo.name)
    return path_list


def try_importing_all_submodules(package):
    modules_with_errors = []
    for m in find_abs_modules(package):
        try:
            importlib.import_module(m)
        except Exception as e:
            modules_with_errors.append(m)
            print(f"{m}: {e.__class__.__name__}: {e}")
    if len(modules_with_errors) > 0:
        raise ImportError(
            f"{len(modules_with_errors)} submodules of {package.__name__} cannot be imported\n"
            + f"{modules_with_errors}"
        )


def test_importing_all_submodules():
    import skdecide

    try_importing_all_submodules(skdecide)


if __name__ == "__main__":
    test_importing_all_submodules()
