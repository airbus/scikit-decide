# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

register_inner_solver(vi "VISolver<Domain, Texecution>" "VI" true
    "has_get_next_state_distribution<Domain>::value && has_is_terminal<Domain>::value")

IF (BUILD_PYTHON_BINDING OR ONLY_PYTHON)
    generate_template_instantiation_files(
        PYTHON_VI_SOLVER_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/py_vi_solver.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "Texecution" "skdecide::SequentialExecution!Seq;skdecide::ParallelExecution!Par")

    ADD_LIBRARY(py_vi STATIC
                ${CMAKE_CURRENT_SOURCE_DIR}/py_vi.cc
                ${PYTHON_VI_SOLVER_TEMPLATE_FILES})
    TARGET_INCLUDE_DIRECTORIES(py_vi PRIVATE ${INCLUDE_DIRS})
    TARGET_LINK_LIBRARIES(py_vi ${LIBS})

    CMAKE_POLICY(SET CMP0079 NEW)
    TARGET_LINK_LIBRARIES(__skdecide_hub_cpp PRIVATE py_vi)
ENDIF ()
