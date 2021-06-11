cmake_minimum_required(VERSION 3.19)
project(cs205_project_simple_matrix)

set(CMAKE_CXX_STANDARD 17)

#specify your opencv path
set(OpenCV_DIR "E:/Environment/opencv")

#specify your eigen path
include_directories("E:\\Environment\\eigen3")

find_package(OpenCV REQUIRED)
IF(OPENCV_FOUND)
    include_directories(
            ${LOCAL_INCLUDE_DIR}
            ${OpenCV_INCLUDE_DIRS}
    )
ELSE()
    message(FATAL_ERROR "Missed OpenCV! CMake will exit")
ENDIF()

add_executable(cs205_project_simple_matrix

        src/simple_matrix.h
        test/test1.cpp
        src/simple_matrix_exception.h
        src/simple_matrix_exception.cpp
        )

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})