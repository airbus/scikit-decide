# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os, sys
import numpy as np
from skdecide.builders.discrete_optimization.generic_tools.do_problem import Problem, EncodingRegister, TypeAttribute


def get_attribute_for_type(problem: Problem, type_attribute: TypeAttribute):
    register = problem.get_attribute_register()
    attributes = [k 
                  for k in register.dict_attribute_to_type
                  for t in register.dict_attribute_to_type[k]["type"]
                  if t == type_attribute]
    return attributes