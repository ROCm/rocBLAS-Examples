# rocBLAS-Examples CMake
This rocBLAS example shows the use of CMake to build an excutable that utilizes rocBLAS.  The main.cpp C++ code illustrates nothing new, and the standard Makefiles provide for the entire rocBLAS-Examples make build system so they can be ignored.  Thus the only files of interest are the CMake files CMakeLists.txt and src/CMakeLists.txt.   The CMake project depends only on the rocBLAS package which automatically brings in the rocBLAS dependencies for hip to allow inclusion of hip header files in main.cpp without explicit include search paths being added.  This example assumes you are already familiar with CMake as it will not explain the structure.

## Documentation
Run the example without any command line arguments to use default values for a matrix size.  Otherwise a single argument runs the geam function with (M=N=argument with alpha=1, beta=2).

    Usage: ./build/src/example-cmake [size]
        [size]          Matrix dimension (default 2048)


## Building
These examples require that you have an installation of rocBLAS on your machine. You do not require sudo or other access to build these examples which default to compile using gcc but can also use the the hipcc compiler from the rocBLAS installation.  If rocBLAS is not installed you can set the environment variable ROCBLAS_PATH to point to the location of your rocblas build.  Note the standard cmake style of building is invoked via make using the file Makefile.cmake.   However, using this pattern yourself you don't need the top level Makefiles and would invoke cmake directly from a build directory, and can specify the compiler using the form: CXX=g++ cmake ..

    cd BuildTools/CMake
    mkdir build
    cd build
    cmake ..
    make
    ./src/example-cmake

