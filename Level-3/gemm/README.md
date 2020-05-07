# rocBLAS-Examples gemm
Example showing moving matrix and vector data to the GPU device and calling the rocblas gemm (general matrix matrix product) function. Results are fetched from GPU and compared against a CPU implementation.

## Documentation
Run the example without any command line arguments to use default values.
Running with --help will show the options:

    Usage: ./gemm
      --K <value>              Matrix/vector dimension
      --M <value>              Matrix/vector dimension
      --N <value>              Matrix/vector dimension
      --alpha <value>          Alpha scalar
      --beta <value>           Beta scalar

## Building
These examples require that you have an installation of rocBLAS on your machine.  You do not required sudo or other access to build these examples which default to compile using gcc but can also use the the hipcc compiler from the rocBLAS installation.   The use of hipcc compiler can be set by uncommenting lines in the Makefiles.  If rocBLAS is not installed you can set the environment variable ROCBLAS_PATH to point to the location of your rocblas build. 

    cd Level-1/gemm 
    make
    ./gemm
 