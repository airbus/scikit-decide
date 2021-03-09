# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

IF (BUILD_PYTHON_BINDING OR ONLY_PYTHON)
    PYBIND11_ADD_MODULE(__skdecide_hub_cpp
                            py_skdecide.cc
                            ${BACKWARD_ENABLE})
    TARGET_INCLUDE_DIRECTORIES(__skdecide_hub_cpp PRIVATE ${INCLUDE_DIRS})
    TARGET_LINK_LIBRARIES(__skdecide_hub_cpp PRIVATE
                            ${LIBS}
                            INTERFACE ${INTERFACES})
ENDIF ()

ADD_SUBDIRECTORY (solver)
