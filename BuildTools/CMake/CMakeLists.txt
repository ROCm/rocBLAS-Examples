
cmake_minimum_required( VERSION 3.5 )

project( example-cmake LANGUAGES CXX)

find_package(rocblas REQUIRED)

add_subdirectory( src )
