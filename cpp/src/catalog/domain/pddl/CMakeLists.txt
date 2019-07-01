ADD_LIBRARY(airlaps_pddl OBJECT
                pddl.cc
                domain.cc
                requirements.cc
                type.cc
                parser.cc)
TARGET_INCLUDE_DIRECTORIES(airlaps_pddl PRIVATE ${INCLUDE_DIRS})

IF (NOT ONLY_PYTHON)
        TARGET_SOURCES(airlaps PRIVATE
                        $<TARGET_OBJECTS:airlaps_pddl>)
ENDIF ()

IF (BUILD_PYTHON_BINDING OR ONLY_PYTHON)
        TARGET_SOURCES(__airlaps PRIVATE
                        $<TARGET_OBJECTS:airlaps_pddl>
                        ${CMAKE_CURRENT_SOURCE_DIR}/py_pddl.cc)
ENDIF ()
