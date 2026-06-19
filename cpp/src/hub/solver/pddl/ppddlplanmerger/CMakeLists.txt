# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

IF (BUILD_PYTHON_BINDING OR ONLY_PYTHON)
    generate_template_instantiation_files(
        PYTHON_PPDDLPLANMERGER_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/py_ppddlplanmerger_solver.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "Texecution" "skdecide::SequentialExecution!Seq;skdecide::ParallelExecution!Par"
        "Tstrategy" "skdecide::AllOutcomesStrategy!All;skdecide::MostProbableOutcomeStrategy!Mpo;skdecide::RandomOutcomeStrategy!Rnd")

    ADD_LIBRARY(py_ppddlplanmerger STATIC
        ${CMAKE_CURRENT_SOURCE_DIR}/py_ppddlplanmerger.cc
        ${PYTHON_PPDDLPLANMERGER_TEMPLATE_FILES})
    TARGET_INCLUDE_DIRECTORIES(py_ppddlplanmerger PRIVATE ${INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/src/hub/domain/pddl)
    TARGET_LINK_LIBRARIES(py_ppddlplanmerger ${LIBS} skdecide_pddl py_ff)

    CMAKE_POLICY(SET CMP0079 NEW)
    TARGET_LINK_LIBRARIES(__skdecide_hub_cpp PRIVATE py_ppddlplanmerger)
ENDIF ()
