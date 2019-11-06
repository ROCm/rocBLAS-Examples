# rocBLAS-Examples gemv
Example showing moving matrix and vector data to the GPU device and calling the rocblas gemv (general matrix vector product) function. Results are fetched from GPU and compared against a CPU implementation and displayed.

## Documentation
Run the example without any command line arguments to use default values.
Running with --help will show the options:

    Usage: ./gemv
      --M <value>              Matrix/vector dimension
      --N <value>              Matrix/vector dimension
      --alpha <value>          Alpha scalar
      --beta <value>           Beta scalar
      --xinc <value>           Increment for x vector
      --yinc <value>           Increment for y vector

## Building
These examples require that you have an installation of rocBLAS on your machine.  You do not required sudo or other access to build these examples which default to compile using gcc but can also use the the hcc compiler from the rocBLAS installation.   The use of hcc compiler can be set by uncommenting lines in the Makefiles.  If rocBLAS is not installed you can set the environment variable ROCBLAS_PATH to point to the location of your rocblas build.

    cd Level-1/gemv 
    make
    ./gemv
 