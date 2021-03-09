# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

LIST(APPEND INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/src)

# IF (NOT ONLY_PYTHON)
#     ADD_LIBRARY(skdecide
#                     ${BACKWARD_ENABLE})
#     TARGET_INCLUDE_DIRECTORIES(skdecide PRIVATE ${INCLUDE_DIRS})
#     TARGET_LINK_LIBRARIES(skdecide PRIVATE
#                             ${LIBS}
#                             INTERFACE ${INTERFACES})
# ENDIF ()

ADD_SUBDIRECTORY(hub)
ADD_SUBDIRECTORY(utils)
