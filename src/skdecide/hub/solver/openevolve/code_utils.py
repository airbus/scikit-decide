"""Utility functions for evolved code."""

import re
import sys
from collections.abc import Iterable
from difflib import unified_diff
from types import ModuleType
from typing import Union

from openevolve.utils.code_utils import parse_evolve_blocks


def extract_code_outside_evolveblocks(code: str) -> str:
    """Extract code outside evolve blocks."""
    code_lines = code.splitlines(True)
    res = parse_evolve_blocks(code)
    out_code = ""
    out_start = 0
    for in_start, in_end, in_block in res:
        out_code += "".join(code_lines[out_start : in_start + 1])
        out_start = in_end
    out_code += "".join(code_lines[out_start:])
    return out_code


def check_diff_outside_evolveblocks(
    code1: str, code2: str, show_diff: bool = True, path1: str = "", path2: str = ""
):
    """Check diff outside evolve blocks

    # Parameters
    code1:
    code2:
    show_diff: print the diff if any
    path1: path corresponding to code1 (for diff printing)
    path2: path corresponding to code2 (for diff printing)

    # Returns
    True if no differences

    """
    out_code1 = extract_code_outside_evolveblocks(code1)
    out_code2 = extract_code_outside_evolveblocks(code2)
    check = out_code1 == out_code2
    if not check:
        print(
            "".join(
                unified_diff(
                    out_code1.splitlines(True),
                    out_code2.splitlines(True),
                    fromfile=path1,
                    tofile=path2,
                )
            )
        )
    return check


def check_diff_outside_evolveblocks_from_paths(path1, path2, show_diff=True):
    """Check diff outside evolve blocks

    # Parameters
    path1:
    path2:
    show_diff: print the diff if any

    # Returns
    True if no differences

    """
    with open(path1, "r") as f:
        code1 = f.read()
    with open(path2, "r") as f:
        code2 = f.read()
    return check_diff_outside_evolveblocks(
        code1, code2, show_diff=show_diff, path1=path1, path2=path2
    )


def get_sorted_imported_modules(
    modules_to_parse: Union[Iterable[ModuleType], ModuleType],
    recursive: bool = True,
    module_filter_pattern: str = "^skdecide\.",
) -> list[ModuleType]:
    """Get recursively the immported modules respecting a pattern.

    # Parameters
    modules_to_parse: list of modules to inspect
    recursive: whether applying the function to imported modules
    module_filter_pattern: pattern that the imported modules need to respect to be included

    # Returns
    List of modules to import from the given modules respecting the given pattern,
    sorted by import depth.

    """
    if isinstance(modules_to_parse, ModuleType):
        modules_to_parse = [modules_to_parse]
    module_filter_built_pattern = re.compile(module_filter_pattern)
    modules_parsed = list()
    modules_to_include = list(modules_to_parse)
    while modules_to_parse:
        module_to_parse = modules_to_parse.pop(0)
        modules_parsed.append(module_to_parse)
        for obj in vars(module_to_parse).values():
            if hasattr(obj, "__module__"):
                new_module_name = obj.__module__
                if module_filter_built_pattern.match(new_module_name):
                    new_module = sys.modules[new_module_name]
                    if new_module not in modules_to_include:
                        modules_to_include.append(new_module)
                    if (
                        recursive
                        and new_module not in modules_parsed
                        and new_module not in modules_to_parse
                    ):
                        modules_to_parse.append(new_module)
    return modules_to_include
