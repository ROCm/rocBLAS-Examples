# rocBLAS-Examples HIP
Example showing HIP program asynchronously moving data to the GPU device and calling a hip kernel for some computation and then the rocblas dgeam function with the results. Device results are fetched from GPU and compared against a CPU implementation and displayed.  This example uses the hipcc for compilation.

## Documentation
Run the example without any command line arguments to use default values.

    Usage: ./HIP [rows] [cols]                Matrix of dimension rows x cols (default 256 x 512)

The make run target shows the use of AMD_LOG_LEVEL environment variable to display the hip API calls being made at runtime.

## Building
These examples require that you have an installation of rocBLAS on your machine.  You do not required sudo or other access to build these examples. This example uses hipcc from the rocBLAS installation. If rocBLAS is not installed you can set the environment variable ROCBLAS_PATH to point to the location of your rocblas build.

    cd Languages/HIP
    make
    ./HIP

