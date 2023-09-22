# rocBLAS-Examples trmv
Example showing moving matrix and vector data to the GPU device and calling the rocblas trmv function which does the matrix-vector operations. hipResults are fetched from GPU and compared against a CPU implementation and displayed. This example illustrates the mixed usage of 2 different complex types with the same memory layout (std::complex<float>, and rocblas_float_complex) and additionally shows absolute and relative error checking.  This example uses the helper::GPUTimer which can be viewed to see how hip API calls can be used to time computation in a stream using events.

## Documentation
Run the example without any command line arguments to use default values.
Running with --help will show the options:

    Usage: ./trmv
      --N <value>              Matrix/vector dimension
      --incx <value>           Increment for x vector

## Building
These examples require that you have an installation of rocBLAS on your machine.  You do not required sudo or other access to build these examples which default to compile using gcc but can also use the the hipcc compiler from the rocBLAS installation.   The use of hipcc compiler can be set by uncommenting lines in the Makefiles.  If rocBLAS is not installed you can set the environment variable ROCBLAS_PATH to point to the location of your rocblas build.

    cd Level-2/trmv
    make
    ./trmv

