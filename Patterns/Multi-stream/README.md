# rocBLAS-Examples Multi-stream 
This example presents an input matrix 'A' which is 'N × N' symmetric matrix stored in upper triangular mode, input vectors 'X', 'Y' of size 'N x incx' and 'N x incy' respectively. They are transferred asynchronously from the host (CPU) to the device (GPU). The 'Alpha' and 'Beta' are scalar values. After calling the rocblas symv (symmetric matrix-vector product) function, the vector 'y' overwritten with the result is transferred asynchronously from the device to the host and compared against a CPU implementation and displayed.  This example uses the helper::GPUTimer to see how hip API calls can be used to time computation in a stream using events.

## Documentation
Run the example without any command line arguments to use default values (N=5, alpha=1, beta=1, incx=1, incy=1).
Running with --help will show the options:

    Usage: ./Multi-stream
      --N <value>              Matrix/vector dimension
      --alpha <value>          Alpha scalar
      --beta <value>           Beta scalar
      --incx <value>           Increment for x vector
      --incy <value>           Increment for y vector

## Building
These examples require that you have an installation of rocBLAS on your machine. You do not require sudo or other access to build these examples which default to compile using gcc but can also use the the hipcc compiler from the rocBLAS installation. The use of hipcc compiler can be set by uncommenting lines in the Makefiles. If rocBLAS is not installed you can set the environment variable ROCBLAS_PATH to point to the location of your rocblas build.

    cd Patterns/Multi-stream 
    make
    ./Multi-stream
 
