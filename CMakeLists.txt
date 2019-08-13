project(nnc)
cmake_minimum_required(VERSION 3.0)
include_directories(
	"${CMAKE_CURRENT_SOURCE_DIR}/inc"
	"${CMAKE_CURRENT_SOURCE_DIR}/libs/include"
)
file(GLOB SRCS
	"${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/layers/*.cc"
)
set(CMAKE_CXX_FLAGS "--std=c++11 -fsigned-char -Wall -Wno-comment -g")
set(PROTOBUF_LIBRARY     "${CMAKE_CURRENT_SOURCE_DIR}/libs/lib/libprotobuf.so")
set(PROTOBUF_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libs/include")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
find_package(Protobuf REQUIRED)


add_custom_command (
	OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/src/caffe.pb.cc"
	OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/inc/caffe.pb.h"
	COMMENT "Compiling proto schema file."
	DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/proto/caffe.proto"
	WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/proto"
	COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/libs/bin/protoc" --cpp_out=./ caffe.proto
	COMMAND mv caffe.pb.h  "${CMAKE_CURRENT_SOURCE_DIR}/inc"
	COMMAND mv caffe.pb.cc "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

add_executable(nnc
	"${SRCS}"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/caffe.pb.cc"
)
target_link_libraries(nnc ${PROTOBUF_LIBRARIES})
