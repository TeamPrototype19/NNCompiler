project(nnc)
cmake_minimum_required(VERISON 3.10)
include_directories(
	"$(CMAKE_CURRENT_SOURCE_DIR}/inc"
)
file(GLOB SRCS
	"${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc"
)
set(CMAKE_FXX_FLAGS "--std=c++11 -m64 -Wall -Wno-comment -g")
FIND_PACKAGE(Protobuf REQUIRED)

add_executable(nnc
	"${SRCS}"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/caffe.pb.cc"
)

add_custom_commnd (
	"${CMAKE_CURRENT_SOURCE_DIR}/src/caffe.pb.cc"
	"${CMAKE_CURRENT_SOURCE_DIR}/inc/caffe.pb.h"
	COMMENT "Compiling proto schema file."
	DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/proto/caffe.proto"
	WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/proto"
	COMMAND protoc --cpp_out=./ caffe.proto
	COMMAND mv caffe.pb.h  "${CMAKE_CURRENT_SOURCE_DIR}/inc"
	COMMAND mv caffe.pb.cc "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_link_libraries(nnc ${PROTOBUF_LIBRARIES})
