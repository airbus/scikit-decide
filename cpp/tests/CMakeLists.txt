SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
                      "${CMAKE_SOURCE_DIR}/sdk/Catch2-${CATCH2_VERSION}/CMake"
                      "${CMAKE_SOURCE_DIR}/sdk/Catch2-${CATCH2_VERSION}/contrib")

INCLUDE(CTest)
INCLUDE(Catch)

LIST(APPEND INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/src)
LIST(APPEND INCLUDE_DIRS ${CATCH2_INCLUDE_DIR})

MACRO(airlaps_test name)
    ADD_EXECUTABLE(test-${name} test_${name}.cc $<TARGET_OBJECTS:tests-main>)
    TARGET_INCLUDE_DIRECTORIES(test-${name} PRIVATE ${INCLUDE_DIRS})
    TARGET_LINK_LIBRARIES(test-${name}
                            PRIVATE Catch2::Catch2 ${LIBS}
                            INTERFACE ${INTERFACES})
    CATCH_DISCOVER_TESTS(test-${name})
ENDMACRO ()

ADD_LIBRARY(tests-main OBJECT tests_main.cc)
TARGET_INCLUDE_DIRECTORIES(tests-main PRIVATE ${CATCH2_INCLUDE_DIR})

airlaps_test(core)
airlaps_test(memory)
airlaps_test(observability)
airlaps_test(events)
airlaps_test(initialization)
airlaps_test(goals)
airlaps_test(dynamics)
airlaps_test(aostar)
