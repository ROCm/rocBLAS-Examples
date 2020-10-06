
# target
add_executable( example-cmake main.cpp )

# rocBLAS-examples common include path, e.g. "error_macros.h"
target_include_directories( example-cmake PRIVATE ../../../common )

target_link_libraries( example-cmake PRIVATE roc::rocblas )
