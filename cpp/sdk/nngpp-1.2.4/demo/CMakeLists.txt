add_executable(async_server async/server.cpp)
target_link_libraries(async_server nngpp)
target_include_directories(async_server PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_executable(async_client async/client.cpp)
target_link_libraries(async_client nngpp)
target_include_directories(async_client PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_executable(http_client http_client/http_client.cpp)
target_link_libraries(http_client nngpp)
target_include_directories(http_client PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_executable(raw raw/raw.cpp)
target_link_libraries(raw nngpp)
target_include_directories(raw PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_executable(reqrep reqrep/reqrep.cpp)
target_link_libraries(reqrep nngpp)
target_include_directories(reqrep PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_executable(rest rest/server.cpp)
target_link_libraries(rest nngpp)
target_include_directories(rest PUBLIC ${CMAKE_SOURCE_DIR}/include)

