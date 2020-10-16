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
 
## Calculation of allowed_error

Individual entries in the result matrix 'c' are calculated using:

  for(int i = 0; i < M; i++)
  {
    for(int j = 0; j < N; j++)
    {
      accumulator = 0;
      for(int k = 0, k < K, k++)
      {
        accumulator += a[i + k * lda] * b[k + j * ldb];
      }
      c[i + j * ldc] = accumulator;
    }
  }


With IEEE arithmetic, the allowed roundoff error for each update of the accumulator in the loop is 0.5 * ULP (unit of least precision). When the magnitude of the accumulator is 1 then ULP is approximately equal to eps (epsilon). When the magnitude of the accumulator is |accumulator|, then ULP is approximately
equal to |accumulator| * eps.

The correctness check requires the calculated result to be close enough to a reference result. The worst-case for the calculated result differing from the reference result would occur if every update of the accumulator in the calculated result rounded up 0.5 ULP and every update of the accumulator in the reference result rounded down 0.5 ULP or vice versa. The matrices a and b have pseudo-random values between 0 and 1. The worst-case for the error between the calculated and reference result would be if every pseudo-random value has the maximum value of 1. This would mean that the accumulator would have values 1, 2, 3, ... K.

If we have both the worst cases above, then the allowable error is approximately:
eps * (1 + 2 + 3 + ... + K) = eps * 0.5 * K * (K+1)
The statistical probability that all 2 * K pseudo-random values in matrices a and b will be equal to 1 is small. The statistical probability that the calculated result will always round up and the reference result will always round down (or vice versa) is small. In place of the K in the above allowable error, the statistical argument suggests we use sqrt(K). Because this is a non-rigorous argument, in place of 0.5 we use a tolerance of 10. Thus, we allow an error between the calculated and reference result of:
 eps * sqrt(K) * K * 10.

