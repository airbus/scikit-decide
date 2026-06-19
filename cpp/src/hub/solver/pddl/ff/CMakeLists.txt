# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

IF (BUILD_PYTHON_BINDING OR ONLY_PYTHON)
    ADD_LIBRARY(py_ff STATIC
        ${CMAKE_CURRENT_SOURCE_DIR}/ff.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/py_ff.cc)
    TARGET_INCLUDE_DIRECTORIES(py_ff PRIVATE ${INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/src/hub/domain/pddl)
    TARGET_LINK_LIBRARIES(py_ff ${LIBS} skdecide_pddl)

    CMAKE_POLICY(SET CMP0079 NEW)
    TARGET_LINK_LIBRARIES(__skdecide_hub_cpp PRIVATE py_ff)
ENDIF ()
