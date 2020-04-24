SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
                      "${CMAKE_SOURCE_DIR}/sdk/Catch2/CMake"
                      "${CMAKE_SOURCE_DIR}/sdk/Catch2/contrib")

INCLUDE(CTest)
INCLUDE(Catch)

LIST(APPEND INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/src)
LIST(APPEND INCLUDE_DIRS ${CATCH2_INCLUDE_DIR})

MACRO(skdecide_test name)
    ADD_EXECUTABLE(test-${name} test_${name}.cc $<TARGET_OBJECTS:tests-main>)
    TARGET_INCLUDE_DIRECTORIES(test-${name} PRIVATE ${INCLUDE_DIRS})
    TARGET_LINK_LIBRARIES(test-${name}
                            PRIVATE Catch2::Catch2 ${LIBS}
                            INTERFACE ${INTERFACES})
    CATCH_DISCOVER_TESTS(test-${name})
ENDMACRO ()

ADD_LIBRARY(tests-main OBJECT tests_main.cc)
TARGET_INCLUDE_DIRECTORIES(tests-main PRIVATE ${CATCH2_INCLUDE_DIR})

skdecide_test(core)
skdecide_test(memory)
skdecide_test(observability)
skdecide_test(events)
skdecide_test(initialization)
skdecide_test(goals)
skdecide_test(dynamics)
