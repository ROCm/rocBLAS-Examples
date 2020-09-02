# rocBLAS-Examples nrm2
This example presents vector 'X' and a scalar 'Result' transferred to the GPU device and calling the rocBLAS nrm2 function. Then, The rocBLAS nrm2 function computes the Euclidean norm of vector 'X' and stores the output in 'Result'. Finally, the gold-standard value 'goldResult' (computed in CPU) along with the Result 'hResult' (Computed in GPU) are displayed for comparison.

## Documentation
Run the example without any command line arguments to use default values (incx=1, n=5).
Running with --help will show the options:

    Usage: ./nrm2
        --incx <value>           Increment for x vector
        --n <value>              Size of vector
        
## Building
These examples require that you have an installation of rocBLAS on your machine. You do not require sudo or other access to build these examples which default to compile using gcc but can also use the the hipcc compiler from the rocBLAS installation. The use of hipcc compiler can be set by uncommenting lines in the Makefiles. If rocBLAS is not installed you can set the environment variable ROCBLAS_PATH to point to the location of your rocblas build. 

    cd Level-1/nrm2
    make
    ./nrm2