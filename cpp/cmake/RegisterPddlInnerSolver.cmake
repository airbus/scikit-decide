# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Macro for registering PDDL inner solvers.
# Each solver calls register_pddl_inner_solver() in its CMakeLists.txt.
# After all ADD_SUBDIRECTORY calls, the registry is generated via configure_file.
#
# Usage:
#   register_pddl_inner_solver(ff_inner_solver
#       "pddl::FFInnerSolver<Texecution>" "FF" ppddlreplan)

set(PDDL_INNER_SOLVERS "" CACHE INTERNAL "List of PDDL inner solver registrations")

macro(register_pddl_inner_solver name class display_name dir)
    set(PDDL_INNER_SOLVERS "${PDDL_INNER_SOLVERS};${name}|${dir}|${class}|${display_name}" CACHE INTERNAL "")
endmacro()
