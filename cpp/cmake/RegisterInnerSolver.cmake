# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Macro for registering C++ solvers as inner solvers.
# Each solver calls register_inner_solver() in its CMakeLists.txt.
# After all ADD_SUBDIRECTORY calls, the registry is generated via configure_file.
#
# Usage:
#   register_inner_solver(aostar "AOStarSolver<Domain, Texecution>" "AOstar" false
#       "has_get_next_state_distribution<Domain>::value")
#   register_inner_solver(ssplp "SSPLPSolver<Domain, Texecution>" "SSPLP" false
#       "has_get_next_state_distribution<Domain>::value && has_is_terminal<Domain>::value" mdplp)

set(INNER_SOLVERS "" CACHE INTERNAL "List of inner solver registrations")

macro(register_inner_solver name class display_name supports_tv requires_expr)
    set(_dir "${name}")
    foreach(_arg ${ARGN})
        set(_dir "${_arg}")
    endforeach()
    set(INNER_SOLVERS "${INNER_SOLVERS};${name}|${_dir}|${class}|${display_name}|${supports_tv}|${requires_expr}" CACHE INTERNAL "")
endmacro()
