# rocBLAS-Examples CMake
This example shows how to use rocBLAS in a C++ program with a CMake build system.  The focus for this example is the setup in [CMakeLists.txt](CMakeLists.txt) and [src/CMakeLists.txt](src/CMakeLists.txt).  The C++ code in `main.cpp` is just an example of some rocBLAS function calls, and the Makefiles are there only as part of the rocBLAS-Examples make build system, so they can be ignored.  The CMake project depends only on the rocBLAS package, which automatically brings in the rocBLAS dependencies for hip. This allows inclusion of hip header files in `main.cpp` without having to explicitly specify include search paths.

## Documentation
Run the example without any command line arguments to use default values for a matrix size.  Otherwise a single argument runs the geam function with (M=N=argument with alpha=1, beta=2).

    Usage: ./build/src/example-cmake [size]
        [size]          Matrix dimension (default 2048)


## Building
These examples require that you have rocBLAS on your machine. If rocBLAS is not installed you can set the environment variable ROCBLAS_PATH to point to the location of your rocBLAS build.  The makefile defaults to compile using g++, but you can also use the the hipcc compiler from the ROCm installation.    Note the standard cmake style of building is invoked via make within the file Makefile-run-cmake.   However, using this pattern yourself you don't need the top level Makefiles and would invoke cmake directly from a build directory, and can specify the compiler using the form: CXX=g++ cmake ..

    cd BuildTools/CMake
    mkdir build
    cd build
    cmake ..
    make
    ./src/example-cmake
