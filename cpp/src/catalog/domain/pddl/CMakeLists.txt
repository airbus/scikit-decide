ADD_LIBRARY(airlaps_pddl OBJECT
                domain.cc
                requirements.cc
                parser.cc)
TARGET_INCLUDE_DIRECTORIES(airlaps_pddl PRIVATE ${INCLUDE_DIRS})

# IF (BUILD_PYTHON_BINDING OR ONLY_PYTHON)
#         TARGET_SOURCES(__airlaps PRIVATE
#                         ${CMAKE_CURRENT_SOURCE_DIR}/py_pddl.cc)
# ENDIF ()
