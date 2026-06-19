# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

register_pddl_inner_solver(ff_inner_solver
    "skdecide::pddl::FFInnerSolver<Texecution>" "FF" ppddlreplan)

IF (BUILD_PYTHON_BINDING OR ONLY_PYTHON)
    generate_template_instantiation_files(
        PYTHON_PPDDLREPLAN_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/py_ppddlreplan_solver.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "Texecution" "skdecide::SequentialExecution!Seq;skdecide::ParallelExecution!Par"
        "Tstrategy" "skdecide::AllOutcomesStrategy!All;skdecide::MostProbableOutcomeStrategy!Mpo;skdecide::RandomOutcomeStrategy!Rnd")

    ADD_LIBRARY(py_ppddlreplan STATIC
        ${CMAKE_CURRENT_SOURCE_DIR}/py_ppddlreplan.cc
        ${PYTHON_PPDDLREPLAN_TEMPLATE_FILES})
    TARGET_INCLUDE_DIRECTORIES(py_ppddlreplan PRIVATE ${INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/src/hub/domain/pddl)
    TARGET_LINK_LIBRARIES(py_ppddlreplan ${LIBS} skdecide_pddl py_ff)

    CMAKE_POLICY(SET CMP0079 NEW)
    TARGET_LINK_LIBRARIES(__skdecide_hub_cpp PRIVATE py_ppddlreplan)
ENDIF ()
