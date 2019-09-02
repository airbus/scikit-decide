IF (BUILD_PYTHON_BINDING OR ONLY_PYTHON)
    PYBIND11_ADD_MODULE(__airlaps_catalog_cpp
                            py_airlaps.cc
                            ${BACKWARD_ENABLE})
    TARGET_INCLUDE_DIRECTORIES(__airlaps_catalog_cpp PRIVATE ${INCLUDE_DIRS})
    TARGET_LINK_LIBRARIES(__airlaps_catalog_cpp PRIVATE
                            ${LIBS}
                            INTERFACE ${INTERFACES})
ENDIF ()

ADD_SUBDIRECTORY (domain)
ADD_SUBDIRECTORY (solver)
