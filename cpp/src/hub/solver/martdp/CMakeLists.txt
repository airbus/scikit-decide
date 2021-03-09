# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

IF (BUILD_PYTHON_BINDING OR ONLY_PYTHON)
    generate_template_instantiation_files(
        PYTHON_MARTDP_SOLVER_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/py_martdp_solver.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "Texecution" "skdecide::SequentialExecution!Seq;skdecide::ParallelExecution!Par")
    
    ADD_LIBRARY(py_martdp STATIC
                ${CMAKE_CURRENT_SOURCE_DIR}/py_martdp.cc
                ${PYTHON_MARTDP_SOLVER_TEMPLATE_FILES})
    TARGET_INCLUDE_DIRECTORIES(py_martdp PRIVATE ${INCLUDE_DIRS})
    TARGET_LINK_LIBRARIES(py_martdp ${LIBS})

    CMAKE_POLICY(SET CMP0079 NEW)
    TARGET_LINK_LIBRARIES(__skdecide_hub_cpp PRIVATE py_martdp)
ENDIF ()
