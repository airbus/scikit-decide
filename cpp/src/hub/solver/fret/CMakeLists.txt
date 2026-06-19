register_inner_solver(fret "FRETSolver<Domain, Texecution, MetaInnerSolverProxy>" "FRET" false
    "has_get_next_state_distribution<Domain>::value && has_is_terminal<Domain>::value")

IF (BUILD_PYTHON_BINDING OR ONLY_PYTHON)
    generate_template_instantiation_files(
        PYTHON_FRET_SOLVER_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/py_fret_solver.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "Texecution" "SequentialExecution!Seq;ParallelExecution!Par")

    ADD_LIBRARY(py_fret STATIC
                ${CMAKE_CURRENT_SOURCE_DIR}/py_fret.cc
                ${PYTHON_FRET_SOLVER_TEMPLATE_FILES})
    TARGET_INCLUDE_DIRECTORIES(py_fret PRIVATE ${INCLUDE_DIRS})
    TARGET_LINK_LIBRARIES(py_fret ${LIBS})

    CMAKE_POLICY(SET CMP0079 NEW)
    TARGET_LINK_LIBRARIES(__skdecide_hub_cpp PRIVATE py_fret)
ENDIF ()
