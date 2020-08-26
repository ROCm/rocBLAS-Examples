# rocBLAS - Examples axpy
This example presents two independent vectors 'X' and 'Y' and scalar value 'alpha' transferred to the GPU device and calling the rocBLAS axpy function. Inside the rocBLAS axpy function, 'alpha' is multiplied with the individual element of vector 'X' and the resultant vector is added with the vector 'Y', overwriting vector 'Y' with the result. Result vector is then retrieved to the host. Then, result vector along with the gold standard (calculated using CPU) are displayed and maximum relative error between them is calculated.

# rocBLAS-Examples axpy
This example presents two independent vectors 'X', 'Y' and scalar 'alpha' transferred to the GPU device and calling the rocBLAS axpy function. Inside the rocBLAS axpy function, 'alpha' is multiplied with the individual element of vector 'X' and the resultant vector is added with the vector 'Y', overwriting vector 'Y' with the result. Result vector is retrieved from the device to the host. Then, result vector along with the gold standard (calculated using CPU) are displayed and maximum relative error between them is calculated.

## Documentation
Run the example without any command line arguments to use default values (alpha=1, incx=1, incy=1, n=5).
Running with --help will show the options:

    Usage: ./axpy
        --alpha <value>          Alpha scalar
        --incx <value>           Increment for x vector
        --incy <value>           Increment for y vector
        --n <value>              Size of vector


## Building
These examples require that you have an installation of rocBLAS on your machine. You do not require sudo or other access to build these examples which default to compile using gcc but can also use the the hipcc compiler from the rocBLAS installation. The use of hipcc compiler can be set by uncommenting lines in the Makefiles. If rocBLAS is not installed you can set the environment variable ROCBLAS_PATH to point to the location of your rocblas build. 

    cd Level-1/axpy
    make
    ./axpy
