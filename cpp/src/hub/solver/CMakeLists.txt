# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Reset inner solver registrations (prevents stale entries across re-configurations)
set(INNER_SOLVERS "" CACHE INTERNAL "")
set(PDDL_INNER_SOLVERS "" CACHE INTERNAL "")

ADD_SUBDIRECTORY(aostar)
ADD_SUBDIRECTORY(astar)
ADD_SUBDIRECTORY(bfws)
ADD_SUBDIRECTORY(ilaostar)
ADD_SUBDIRECTORY(iw)
ADD_SUBDIRECTORY(lrtdp)
ADD_SUBDIRECTORY(martdp)
ADD_SUBDIRECTORY(mcts)
ADD_SUBDIRECTORY(riw)
ADD_SUBDIRECTORY(vi)
ADD_SUBDIRECTORY(pi)
ADD_SUBDIRECTORY(ldfs)
ADD_SUBDIRECTORY(rtdp_bel)
ADD_SUBDIRECTORY(ssipp)
ADD_SUBDIRECTORY(fret)
ADD_SUBDIRECTORY(gpci)
ADD_SUBDIRECTORY(sarsop)
ADD_SUBDIRECTORY(hsvi)
ADD_SUBDIRECTORY(despot)
ADD_SUBDIRECTORY(ehc)
ADD_SUBDIRECTORY(pddl)
ADD_SUBDIRECTORY(sspreplan)
ADD_SUBDIRECTORY(sspdethindsight)
ADD_SUBDIRECTORY(sspplanmerger)
ADD_SUBDIRECTORY(pomcp)
IF (HAS_HIGHS)
    ADD_SUBDIRECTORY(mdplp)
    ADD_SUBDIRECTORY(idual)
    ADD_SUBDIRECTORY(witness)
ENDIF()

# --- Generate inner solver registry from accumulated registrations ---
IF (BUILD_PYTHON_BINDING OR ONLY_PYTHON)
    set(ENTRY_INCLUDES "")
    set(ENTRY_CONDITIONALS "")
    foreach(entry IN LISTS INNER_SOLVERS)
        if(entry)
            string(REPLACE "|" ";" parts "${entry}")
            list(GET parts 0 name)
            list(GET parts 1 dir)
            list(GET parts 2 class)
            list(GET parts 3 display_name)
            list(GET parts 4 supports_tv)
            list(GET parts 5 requires_expr)
            string(APPEND ENTRY_INCLUDES
                "#include \"hub/solver/${dir}/${dir}.hh\"\n#include \"hub/solver/${dir}/impl/${dir}_impl.hh\"\n")
            string(APPEND ENTRY_CONDITIONALS
                "  if constexpr (${requires_expr})\n    registry.push_back(make_entry<${class}, Domain, Texecution>(\"${display_name}\", ${supports_tv}));\n")
        endif()
    endforeach()

    configure_file(
        "${CMAKE_SOURCE_DIR}/src/hub/solver/inner_solver/impl/inner_solver_registry_impl.hh.in"
        "${CMAKE_BINARY_DIR}/src/hub/solver/inner_solver/impl/inner_solver_registry_impl.hh"
        @ONLY)
ENDIF()
