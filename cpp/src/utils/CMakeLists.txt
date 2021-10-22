# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

IF (BUILD_PYTHON_BINDING OR ONLY_PYTHON)
    generate_template_instantiation_files(
        PYTHON_DOMAIN_PROXY_BASE_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/python_domain_proxy_base.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "Texecution" "SequentialExecution!Seq;ParallelExecution!Par")

    generate_template_instantiation_files(
        PYTHON_DOMAIN_PROXY_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/python_domain_proxy.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "Texecution" "SequentialExecution!Seq;ParallelExecution!Par"
        "Tagent" "SingleAgent!Sin;MultiAgent!Mul"
        "Tobservability" "PartiallyObservable!Pobs;FullyObservable!FObs"
        "Tcontrollability" "PartiallyControllable!Pcon;FullyControllable!Fcon"
        "Tmemory" "Markovian!Mar;History!His")

    generate_template_instantiation_files(
        PYTHON_CONTAINER_PROXY_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/python_container_proxy.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "Texecution" "SequentialExecution!Seq;ParallelExecution!Par")

    generate_template_instantiation_files(
        PYTHON_HASH_EQ_TEMPLATE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/impl/python_hash_eq.cc.in"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "Texecution" "SequentialExecution!Seq;ParallelExecution!Par")

    ADD_LIBRARY(py_utils STATIC
                ${CMAKE_CURRENT_SOURCE_DIR}/impl/logging_impl.cc
                ${PYTHON_DOMAIN_PROXY_BASE_TEMPLATE_FILES}
                ${PYTHON_DOMAIN_PROXY_TEMPLATE_FILES}
                ${PYTHON_CONTAINER_PROXY_TEMPLATE_FILES}
                ${PYTHON_HASH_EQ_TEMPLATE_FILES})
    TARGET_INCLUDE_DIRECTORIES(py_utils PRIVATE ${INCLUDE_DIRS})
    TARGET_LINK_LIBRARIES(py_utils ${LIBS})

    CMAKE_POLICY(SET CMP0079 NEW)
    TARGET_LINK_LIBRARIES(__skdecide_hub_cpp PRIVATE py_utils)
ENDIF ()
