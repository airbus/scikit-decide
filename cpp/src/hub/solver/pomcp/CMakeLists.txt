IF (BUILD_PYTHON_BINDING OR ONLY_PYTHON)
    generate_template_instantiation_files(
        PYTHON_POMCP_SOLVER_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/py_pomcp_solver.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "Texecution" "skdecide::SequentialExecution!Seq;skdecide::ParallelExecution!Par")

    ADD_LIBRARY(py_pomcp STATIC
                ${CMAKE_CURRENT_SOURCE_DIR}/py_pomcp.cc
                ${PYTHON_POMCP_SOLVER_TEMPLATE_FILES})
    TARGET_INCLUDE_DIRECTORIES(py_pomcp PRIVATE ${INCLUDE_DIRS})
    TARGET_LINK_LIBRARIES(py_pomcp ${LIBS})

    CMAKE_POLICY(SET CMP0079 NEW)
    TARGET_LINK_LIBRARIES(__skdecide_hub_cpp PRIVATE py_pomcp)
ENDIF ()
