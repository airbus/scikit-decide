register_inner_solver(ssipp "SSiPPSolver<Domain, Texecution, MetaInnerSolverProxy>" "SSiPP" false
    "has_get_next_state_distribution<Domain>::value")

IF (BUILD_PYTHON_BINDING OR ONLY_PYTHON)
    generate_template_instantiation_files(
        PYTHON_SSIPP_SOLVER_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/py_ssipp_solver.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "Texecution" "SequentialExecution!Seq;ParallelExecution!Par")

    ADD_LIBRARY(py_ssipp STATIC
                ${CMAKE_CURRENT_SOURCE_DIR}/py_ssipp.cc
                ${PYTHON_SSIPP_SOLVER_TEMPLATE_FILES})
    TARGET_INCLUDE_DIRECTORIES(py_ssipp PRIVATE ${INCLUDE_DIRS})
    TARGET_LINK_LIBRARIES(py_ssipp ${LIBS})

    CMAKE_POLICY(SET CMP0079 NEW)
    TARGET_LINK_LIBRARIES(__skdecide_hub_cpp PRIVATE py_ssipp)
ENDIF ()
