# rocBLAS-Examples Multi-device 
This rocBLAS example showcases the use of multi-devices and multiple streams per device to call the 'rocblas_sgemm' function. Here, the input matrices are 'A' which is an 'M × K' matrix stored in rocblas_operation_none (No Transpose) mode and 'B' which is a 'K × N' matrix stored in rocblas_operation_none (No Transpose) mode. They are allocated per device per stream and transferred asynchronously from the host (CPU) to the device (GPU). The 'Alpha' and 'Beta' are scalar values. After calling the 'rocblas_sgemm' (general matrix-matrix multiply) function the result matrix 'C' which is of size 'M x N' is transferred asynchronously from the device to the host and compared against a CPU implementation. This example uses the helper::GPUTimer, which measures the time taken in the streams using events.

## Documentation
Run the example without any command line arguments to use default values (K=5, M=5, N=5, alpha=1, beta=1).
Running with --help will show the options:

    Usage: ./Multi-device 
      --K <value>              Matrix dimension
      --M <value>              Matrix dimension
      --N <value>              Matrix dimension
      --alpha <value>          Alpha scalar
      --beta <value>           Beta scalar

## Building
These examples require that you have an installation of rocBLAS on your machine. You do not require sudo or other access to build these examples which default to compile using gcc but can also use the the hipcc compiler from the rocBLAS installation. The use of hipcc compiler can be set by uncommenting lines in the Makefiles. If rocBLAS is not installed you can set the environment variable ROCBLAS_PATH to point to the location of your rocblas build.

    cd Patterns/Multi-device 
    make
    ./Multi-device
 
