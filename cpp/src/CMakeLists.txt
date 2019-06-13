SET(INCLUDE_DIRS
        ${CMAKE_SOURCE_DIR}/src
        ${PYBIND11_INCLUDE_DIR}
        ${JSON_INCLUDE_DIR}
        ${PEGTL_INCLUDE_DIR}
        ${SPDLOG_INCLUDE_DIR}
        ${PYTHON_INCLUDE_DIRS})

IF (NOT ONLY_PYTHON)
    ADD_LIBRARY(airlaps
                    ${BACKWARD_ENABLE})
    TARGET_INCLUDE_DIRECTORIES(airlaps PRIVATE ${INCLUDE_DIRS})
    TARGET_LINK_LIBRARIES(airlaps PRIVATE
                            $<TARGET_OBJECTS:airlaps_pddl>
                            spdlog::spdlog)
    ADD_BACKWARD(airlaps)
ENDIF ()

IF (BUILD_PYTHON_BINDING OR ONLY_PYTHON)
    PYBIND11_ADD_MODULE(__airlaps
                            py_airlaps.cc
                            ${BACKWARD_ENABLE})
    TARGET_INCLUDE_DIRECTORIES(__airlaps PRIVATE ${INCLUDE_DIRS})
    TARGET_LINK_LIBRARIES(__airlaps PRIVATE
                            #$<TARGET_OBJECTS:airlaps_pddl>
                            spdlog::spdlog)
    ADD_BACKWARD(__airlaps)
ENDIF ()

ADD_SUBDIRECTORY(catalog)
