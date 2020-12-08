
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