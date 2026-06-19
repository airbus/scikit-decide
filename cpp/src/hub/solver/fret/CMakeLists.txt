IF (BUILD_PYTHON_BINDING OR ONLY_PYTHON)
    generate_template_instantiation_files(
        PYTHON_FRET_SOLVER_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/py_fret_solver.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "Texecution" "SequentialExecution!Seq;ParallelExecution!Par"
        "TinnerSolver" "LRTDPSolver!Lrt;LDFSSolver!Ldf;VISolver!Vi")

    ADD_LIBRARY(py_fret STATIC
                ${CMAKE_CURRENT_SOURCE_DIR}/py_fret.cc
                ${PYTHON_FRET_SOLVER_TEMPLATE_FILES})
    TARGET_INCLUDE_DIRECTORIES(py_fret PRIVATE ${INCLUDE_DIRS})
    TARGET_LINK_LIBRARIES(py_fret ${LIBS})

    CMAKE_POLICY(SET CMP0079 NEW)
    TARGET_LINK_LIBRARIES(__skdecide_hub_cpp PRIVATE py_fret)
ENDIF ()
