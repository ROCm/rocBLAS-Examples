# rocBLAS-Examples gemm_ex_i8_i32_r
Example showing moving 8-bit integer and 32-bit integer matrix data types to the GPU device and calling the rocblas gemm_ex (general matrix matrix product) function. Results are fetched from GPU and compared against a CPU implementation and displayed.  

## Documentation
Run the example without any command line arguments to use default values.
Running with --help will show the options:

    Usage: ./gemm_ex_i8_i32_r
      --K <value>              Matrix/vector dimension
      --M <value>              Matrix/vector dimension
      --N <value>              Matrix/vector dimension
      --alpha <value>          Alpha scalar
      --beta <value>           Beta scalar

## Building
These examples require that you have an installation of rocBLAS on your machine.  You do not required sudo or other access to build these examples which default to compile using gcc but can also use the the hipcc compiler from the rocBLAS installation.   The use of hipcc compiler can be set by uncommenting lines in the Makefiles.  If rocBLAS is not installed you can set the environment variable ROCBLAS_PATH to point to the location of your rocblas build.

    cd Extensions/gemm_ex_i8_i32_r
    make
    ./gemm_ex_i8_i32_r
