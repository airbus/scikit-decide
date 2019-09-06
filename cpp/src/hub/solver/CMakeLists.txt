IF (BUILD_PYTHON_BINDING OR ONLY_PYTHON)
        TARGET_SOURCES(__airlaps_hub_cpp PRIVATE
                        ${CMAKE_CURRENT_SOURCE_DIR}/py_astar.cc
                        ${CMAKE_CURRENT_SOURCE_DIR}/py_aostar.cc
                        ${CMAKE_CURRENT_SOURCE_DIR}/py_iw.cc
                        ${CMAKE_CURRENT_SOURCE_DIR}/py_bfws.cc)
ENDIF ()
