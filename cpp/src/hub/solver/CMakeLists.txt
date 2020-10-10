IF (BUILD_PYTHON_BINDING OR ONLY_PYTHON)
        TARGET_SOURCES(__skdecide_hub_cpp PRIVATE
                        ${CMAKE_CURRENT_SOURCE_DIR}/py_astar.cc
                        ${CMAKE_CURRENT_SOURCE_DIR}/py_aostar.cc
                        ${CMAKE_CURRENT_SOURCE_DIR}/py_mcts.cc
                        ${CMAKE_CURRENT_SOURCE_DIR}/py_iw.cc
                        ${CMAKE_CURRENT_SOURCE_DIR}/py_riw.cc
                        ${CMAKE_CURRENT_SOURCE_DIR}/py_bfws.cc
                        ${CMAKE_CURRENT_SOURCE_DIR}/py_lrtdp.cc
                        ${CMAKE_CURRENT_SOURCE_DIR}/py_ilaostar.cc)
ENDIF ()
