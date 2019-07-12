IF (BUILD_PYTHON_BINDING OR ONLY_PYTHON)
        TARGET_SOURCES(__airlaps PRIVATE
                        ${CMAKE_CURRENT_SOURCE_DIR}/py_aostar.cc
                        ${CMAKE_CURRENT_SOURCE_DIR}/py_iw.cc
                        ${CMAKE_CURRENT_SOURCE_DIR}/gym_spaces.cc)
ENDIF ()
