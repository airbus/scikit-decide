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
