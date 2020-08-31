# rocBLAS-Examples dot
This example presents independent vectors 'X', 'Y' and a scalar value 'Result' transferred to the GPU device and calling the rocBLAS dot function.Then, The rocBLAS dot function computes the dot product of vectors 'X' and 'Y' and stores the output in 'Result'. Next step, transfer the 'Result' from device to host. Finally, the gold-standard value 'goldResult' (computed in CPU) along with the Result 'hResult' (Computed in GPU) are displayed for comparison.

## Documentation
Run the example without any command line arguments to use default values (incx=1, incy=1, n=5).
Running with --help will show the options:

    Usage: ./dot
        --incx <value>           Increment for x vector
        --incy <value>           Increment for y vector
        --n <value>              Size of vector


## Building
These examples require that you have an installation of rocBLAS on your machine.  You do not required sudo or other access to build these examples which default to compile using gcc but can also use the the hipcc compiler from the rocBLAS installation.   The use of hipcc compiler can be set by uncommenting lines in the Makefiles.  If rocBLAS is not installed you can set the environment variable ROCBLAS_PATH to point to the location of your rocblas build. 

    cd Level-1/dot
    make
    ./dot