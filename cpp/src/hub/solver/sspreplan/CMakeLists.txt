IF (BUILD_PYTHON_BINDING OR ONLY_PYTHON)
    generate_template_instantiation_files(
        PYTHON_SSPREPLAN_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/py_sspreplan_solver.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "Texecution" "SequentialExecution!Seq;ParallelExecution!Par"
        "Tstrategy" "AllOutcomesStrategy!All;MostProbableOutcomeStrategy!Mpo;RandomOutcomeStrategy!Rnd")

    ADD_LIBRARY(py_sspreplan STATIC
                ${CMAKE_CURRENT_SOURCE_DIR}/py_sspreplan.cc
                ${PYTHON_SSPREPLAN_TEMPLATE_FILES})
    TARGET_INCLUDE_DIRECTORIES(py_sspreplan PRIVATE ${INCLUDE_DIRS})
    TARGET_LINK_LIBRARIES(py_sspreplan ${LIBS})

    CMAKE_POLICY(SET CMP0079 NEW)
    TARGET_LINK_LIBRARIES(__skdecide_hub_cpp PRIVATE py_sspreplan)
ENDIF ()
