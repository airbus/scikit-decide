ENABLE_LANGUAGE(CXX)
SET(CMAKE_CXX_STANDARD 14)

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
                      "${CMAKE_SOURCE_DIR}/sdk/Catch2-${CATCH2_VERSION}/CMake"
                      "${CMAKE_SOURCE_DIR}/sdk/Catch2-${CATCH2_VERSION}/contrib")

INCLUDE(CTest)
INCLUDE(Catch)

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/src
                    ${JSON_INCLUDE_DIR}
                    ${PEGTL_INCLUDE_DIR}
                    ${SPDLOG_INCLUDE_DIR}
                    ${CATCH2_INCLUDE_DIR})

ADD_LIBRARY(tests-main OBJECT tests_main.cc)

ADD_EXECUTABLE(test-core test_core.cc $<TARGET_OBJECTS:tests-main>)
TARGET_LINK_LIBRARIES(test-core Catch2::Catch2)
CATCH_DISCOVER_TESTS(test-core)

ADD_EXECUTABLE(test-memory test_memory.cc $<TARGET_OBJECTS:tests-main>)
TARGET_LINK_LIBRARIES(test-memory Catch2::Catch2)
CATCH_DISCOVER_TESTS(test-memory)

ADD_EXECUTABLE(test-observability test_observability.cc $<TARGET_OBJECTS:tests-main>)
TARGET_LINK_LIBRARIES(test-observability Catch2::Catch2)
CATCH_DISCOVER_TESTS(test-observability)

ADD_EXECUTABLE(test-events test_events.cc $<TARGET_OBJECTS:tests-main>)
TARGET_LINK_LIBRARIES(test-events Catch2::Catch2)
CATCH_DISCOVER_TESTS(test-events)

ADD_EXECUTABLE(test-initialization test_initialization.cc $<TARGET_OBJECTS:tests-main>)
TARGET_LINK_LIBRARIES(test-initialization Catch2::Catch2)
CATCH_DISCOVER_TESTS(test-initialization)

ADD_EXECUTABLE(test-goals test_goals.cc $<TARGET_OBJECTS:tests-main>)
TARGET_LINK_LIBRARIES(test-goals Catch2::Catch2)
CATCH_DISCOVER_TESTS(test-goals)

ADD_EXECUTABLE(test-dynamics test_dynamics.cc $<TARGET_OBJECTS:tests-main>)
TARGET_LINK_LIBRARIES(test-dynamics Catch2::Catch2)
CATCH_DISCOVER_TESTS(test-dynamics)

ADD_EXECUTABLE(test-aostar test_aostar.cc $<TARGET_OBJECTS:tests-main>)
TARGET_LINK_LIBRARIES(test-aostar Catch2::Catch2)
CATCH_DISCOVER_TESTS(test-aostar)
