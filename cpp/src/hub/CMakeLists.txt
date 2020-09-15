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
