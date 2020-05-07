# rocBLAS-Examples scal
Example showing moving vector data to the GPU device and calling the rocblas scal function. Results are retrieved to host and displayed.

## Documentation
Run the example without any command line arguments to use default values.
Running with --help will show the options:

    Usage: ./scal
      --alpha <value>          Alpha scalar
      --n <value>              Size of vector
      --xinc <value>           Increment for x vector

## Building
These examples require that you have an installation of rocBLAS on your machine.  You do not required sudo or other access to build these examples which default to compile using gcc but can also use the the hipcc compiler from the rocBLAS installation.   The use of hipcc compiler can be set by uncommenting lines in the Makefiles.  If rocBLAS is not installed you can set the environment variable ROCBLAS_PATH to point to the location of your rocblas build. 

    cd Level-1/scal 
    make
    ./scal
 