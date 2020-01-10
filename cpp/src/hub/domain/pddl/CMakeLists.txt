ADD_LIBRARY(skdecide_pddl OBJECT
                pddl.cc
                domain.cc
                requirements.cc
                type.cc
                parser.cc)
TARGET_INCLUDE_DIRECTORIES(skdecide_pddl PRIVATE ${INCLUDE_DIRS})

IF (NOT ONLY_PYTHON)
        TARGET_SOURCES(skdecide PRIVATE
                        $<TARGET_OBJECTS:skdecide_pddl>)
ENDIF ()

IF (BUILD_PYTHON_BINDING OR ONLY_PYTHON)
        TARGET_SOURCES(__skdecide_hub_cpp PRIVATE
                        $<TARGET_OBJECTS:skdecide_pddl>
                        ${CMAKE_CURRENT_SOURCE_DIR}/py_pddl.cc)
ENDIF ()
