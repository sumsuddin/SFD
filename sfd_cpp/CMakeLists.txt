cmake_minimum_required(VERSION 3.1)

set (CMAKE_CXX_STANDARD 11)

find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS} include)
add_definitions(${Caffe_DEFINITIONS})    # ex. -DCPU_ONLY

add_executable(sfd sfd.cpp detection.cpp)
target_link_libraries(sfd ${Caffe_LIBRARIES}) 
