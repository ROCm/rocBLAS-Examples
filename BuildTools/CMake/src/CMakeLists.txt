
# target
add_executable( example-cmake main.cpp )

# include path for example target
target_include_directories( example-cmake PRIVATE ../../../common )

target_link_libraries( example-cmake PRIVATE roc::rocblas )
