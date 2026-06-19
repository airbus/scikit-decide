register_inner_solver(idual "IDualSolver<Domain, Texecution>" "IDual" true
    "has_get_next_state_distribution<Domain>::value && has_is_terminal<Domain>::value")

IF (BUILD_PYTHON_BINDING OR ONLY_PYTHON)
    generate_template_instantiation_files(
        PYTHON_IDUAL_SOLVER_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/py_idual_solver.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "Texecution" "SequentialExecution!Seq;ParallelExecution!Par")

    ADD_LIBRARY(py_idual STATIC
                ${CMAKE_CURRENT_SOURCE_DIR}/py_idual.cc
                ${PYTHON_IDUAL_SOLVER_TEMPLATE_FILES})
    TARGET_INCLUDE_DIRECTORIES(py_idual PRIVATE ${INCLUDE_DIRS}
                               ${highs_SOURCE_DIR}/src)
    TARGET_LINK_LIBRARIES(py_idual ${LIBS} highs)

    CMAKE_POLICY(SET CMP0079 NEW)
    TARGET_LINK_LIBRARIES(__skdecide_hub_cpp PRIVATE py_idual)
ENDIF ()
