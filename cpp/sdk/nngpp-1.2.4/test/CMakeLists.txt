find_package(Catch2)

set(test_sources
    bus.cpp
    device.cpp
    hello_world.cpp
    http_client.cpp
    ipcsupp.cpp
    main.cpp
    message.cpp
    pipe.cpp
    reconnect.cpp
    respondpoll.cpp
    scalability.cpp
    synch.cpp
    tcpsupp.cpp
)

if (NNGPP_BUILD_TLS_TEST)
    list(APPEND test_sources tls.cpp)
endif()

add_executable(tests ${test_sources})
target_include_directories(tests PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(tests nngpp Catch2::Catch2)
