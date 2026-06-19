# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

ADD_SUBDIRECTORY(ff)
ADD_SUBDIRECTORY(ppddlreplan)
ADD_SUBDIRECTORY(ppddldethindsight)
ADD_SUBDIRECTORY(ppddlplanmerger)

# --- Generate PDDL inner solver registry from accumulated registrations ---
IF (BUILD_PYTHON_BINDING OR ONLY_PYTHON)
    set(PDDL_ENTRY_INCLUDES "")
    set(PDDL_ENTRY_LINES "")
    foreach(entry IN LISTS PDDL_INNER_SOLVERS)
        if(entry)
            string(REPLACE "|" ";" parts "${entry}")
            list(GET parts 0 name)
            list(GET parts 1 dir)
            list(GET parts 2 class)
            list(GET parts 3 display_name)
            string(APPEND PDDL_ENTRY_INCLUDES
                "#include \"hub/solver/pddl/${dir}/${name}.hh\"\n#include \"hub/solver/pddl/${dir}/impl/${name}_impl.hh\"\n")
            string(APPEND PDDL_ENTRY_LINES
                "  registry.push_back(make_pddl_entry<${class}, Texecution>(\"${display_name}\"));\n")
        endif()
    endforeach()

    configure_file(
        "${CMAKE_SOURCE_DIR}/src/hub/solver/pddl/inner_solver/impl/pddl_inner_solver_registry_impl.hh.in"
        "${CMAKE_BINARY_DIR}/src/hub/solver/pddl/inner_solver/impl/pddl_inner_solver_registry_impl.hh"
        @ONLY)
ENDIF()
