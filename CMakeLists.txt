cmake_minimum_required(VERSION 3.19)
project(cs205_project_simple_matrix)

set(CMAKE_CXX_STANDARD 17)

include_directories(.)
include_directories(src)

add_executable(cs205_project_simple_matrix

        src/simple_matrix.h
        src/test1.cpp
        src/simple_matrix_exception.h
        src/simple_matrix_exception.cpp
        )
