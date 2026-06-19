IF (BUILD_PYTHON_BINDING OR ONLY_PYTHON)
    generate_template_instantiation_files(
        PYTHON_MDPLP_SOLVER_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/py_mdplp_solver.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "Texecution" "SequentialExecution!Seq;ParallelExecution!Par")

    ADD_LIBRARY(py_mdplp STATIC
                ${CMAKE_CURRENT_SOURCE_DIR}/py_mdplp.cc
                ${PYTHON_MDPLP_SOLVER_TEMPLATE_FILES})
    TARGET_INCLUDE_DIRECTORIES(py_mdplp PRIVATE ${INCLUDE_DIRS}
                               ${highs_SOURCE_DIR}/src)
    TARGET_LINK_LIBRARIES(py_mdplp ${LIBS} highs)

    CMAKE_POLICY(SET CMP0079 NEW)
    TARGET_LINK_LIBRARIES(__skdecide_hub_cpp PRIVATE py_mdplp)
ENDIF ()
