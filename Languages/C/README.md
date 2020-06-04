# rocBLAS-Examples C
Example showing C program asynchronously moving data to the GPU device and calling the rocblas dgeam function. Results are fetched from GPU and compared against a CPU implementation and displayed.  This example uses the gcc -c11 for compilation.

## Documentation
Run the example without any command line arguments to use default values.

    Usage: ./C [N]                 Matrix of NxN dimensions


## Building
These examples require that you have an installation of rocBLAS on your machine.  You do not required sudo or other access to build these examples which default to compile using gcc but can also use the the hipcc compiler from the rocBLAS installation.   The use of hipcc compiler can be set by uncommenting lines in the Makefiles.  This example uses gcc. If rocBLAS is not installed you can set the environment variable ROCBLAS_PATH to point to the location of your rocblas build.

    cd Languages/C
    make
    ./C
 
