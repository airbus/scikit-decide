IF (BUILD_PYTHON_BINDING OR ONLY_PYTHON)
    generate_template_instantiation_files(
        PYTHON_DESPOT_SOLVER_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/py_despot_solver.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "Texecution" "skdecide::SequentialExecution!Seq;skdecide::ParallelExecution!Par")

    ADD_LIBRARY(py_despot STATIC
                ${CMAKE_CURRENT_SOURCE_DIR}/py_despot.cc
                ${PYTHON_DESPOT_SOLVER_TEMPLATE_FILES})
    TARGET_INCLUDE_DIRECTORIES(py_despot PRIVATE ${INCLUDE_DIRS})
    TARGET_LINK_LIBRARIES(py_despot ${LIBS})

    CMAKE_POLICY(SET CMP0079 NEW)
    TARGET_LINK_LIBRARIES(__skdecide_hub_cpp PRIVATE py_despot)
ENDIF ()
