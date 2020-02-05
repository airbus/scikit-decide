cmake_minimum_required (VERSION 3.1)
project(nngpp CXX)

include(GNUInstallDirs)

option(NNGPP_BUILD_DEMOS "build nngpp demos" off)
option(NNGPP_BUILD_TESTS "build nngpp tests" off)

set(CMAKE_CXX_STANDARD 14)

find_package(nng)

file(GLOB_RECURSE includes include/*.h)

add_library(nngpp INTERFACE)
target_include_directories(nngpp INTERFACE $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(nngpp INTERFACE nng::nng)

if (NNGPP_BUILD_DEMOS)
    add_subdirectory(demo)
endif()

if (NNGPP_BUILD_TESTS)
    option(NNGPP_BUILD_TLS_TEST "build nngpp tls tests (requires mbedtls)" off)
    add_subdirectory(test)
endif()

install(TARGETS nngpp EXPORT nngpp-target)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT nngpp-target DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/nngpp
        NAMESPACE nng:: FILE nngpp-config.cmake)
