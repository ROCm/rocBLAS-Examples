/* ************************************************************************
 * Copyright (C) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#include "helpers.hpp"
#include <complex>
#include <hip/hip_runtime.h>
#include <math.h>
#include <rocblas/rocblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

// lda and incx promoted to 64bit to avoid int32 overflow
template <typename T>
void referenceTrmvCalc(rocblas_fill    uplo,
                       std::vector<T>& A,
                       rocblas_int     N,
                       size_t          lda,
                       std::vector<T>& x,
                       std::vector<T>& cpu_ref_result,
                       ssize_t         incx)
{
    // calculate expected result using CPU
    if(uplo == rocblas_fill_lower)
    {
        for(int row = 0; row < N; row++)
        {
            T elem = T(0.0);
            for(int col = 0; col < row + 1; col++)
            {
                elem += A[col * lda + row] * x[col * incx];
            }
            cpu_ref_result[row * incx] = elem;
        }
    }
    else
    {
        for(int row = 0; row < N; row++)
        {
            T elem = T(0.0);
            for(int col = row; col < N; col++)
            {
                elem += A[col * lda + row] * x[col * incx];
            }
            cpu_ref_result[row * incx] = elem;
        }
    }
}

int main(int argc, char** argv)
{

    typedef std::complex<float> T;

    helpers::ArgParser options("Nx");
    if(!options.validArgs(argc, argv))
        return EXIT_FAILURE;

    rocblas_status rstatus = rocblas_status_success;

    rocblas_int N    = options.N;
    rocblas_int incx = options.incx;

    // Pre-filled parameters
    const rocblas_fill     uplo = rocblas_fill_lower;
    const rocblas_diagonal diag = rocblas_diagonal_non_unit;

    //trans is fixed to rocblas_operation_none in this example and support for other options would be added in the future release
    const rocblas_operation trans = rocblas_operation_none;

    size_t      sizeX, sizeA;
    rocblas_int absIncx;

    rocblas_int lda = N;
    absIncx         = incx >= 0 ? incx : -incx;
    sizeX           = size_t(N) * absIncx;
    sizeA           = size_t(lda) * N;

    // Naming: dA is in GPU (device) memory. hA is in CPU (host) memory

    // we are using std::complex for it's operators and it has same memory layout
    // as rocblas_float_complex so can copy the data into the array for use in the rocblas C API
    std::vector<T> hA(sizeA);
    std::vector<T> hX(sizeX);
    std::vector<T> hXCopy(sizeX);
    std::vector<T> hXGold(sizeX);

    // initialize uniform random data with lower and upper range
    helpers::fillVectorUniformRealDist(hA, -0.5, 0.5);
    helpers::fillVectorUniformRealDist(hX, -0.5, 0.5);

    hXCopy = hX;

    //zero out lower/upper part of the matrix depending upon the uplo parameter
    helpers::makeMatrixUpperOrlower(uplo, hA, N, lda);

    //Make matrix unit diagonal depending upon the diag parameter
    if(diag == rocblas_diagonal_unit)
        helpers::make_unit_diagonal(uplo, hA, N, lda);

    // using rocblas API
    rocblas_handle handle;
    rstatus = rocblas_create_handle(&handle);
    CHECK_ROCBLAS_STATUS(rstatus);

    {
        // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory

        // allocate memory on device
        helpers::DeviceVector<rocblas_float_complex> dA(sizeA);
        helpers::DeviceVector<rocblas_float_complex> dX(sizeX);

        if((!dA && sizeA) || (!dX && sizeX))
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return EXIT_FAILURE;
        }

        // time data to device, computation, and data from device back to host
        helpers::GPUTimer gpuTimer;
        gpuTimer.start();

        // copy data from CPU to device (all 3 complex types same memory layout)
        CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * sizeA, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dX, hX.data(), sizeof(T) * sizeX, hipMemcpyHostToDevice));

        // enable passing alpha and beta parameters from pointer to host memory
        rstatus = rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
        CHECK_ROCBLAS_STATUS(rstatus);

        // asynchronous calculation on device, returns before finished calculations
        rstatus = rocblas_ctrmv(handle, uplo, trans, diag, N, dA, lda, dX, incx);

        // check that calculation was launched correctly on device, not that result
        // was computed yet
        CHECK_ROCBLAS_STATUS(rstatus);

        // fetch device memory results, automatically blocked until results ready
        CHECK_HIP_ERROR(
            hipMemcpy(hX.data(), dX, sizeof(rocblas_float_complex) * sizeX, hipMemcpyDeviceToHost));

        gpuTimer.stop();

    } // release device memory via helpers::DeviceVector destructors

    std::cout << "N, lda, incx = " << N << ", " << lda << ", " << incx << std::endl;

    // calculate expected result using CPU
    referenceTrmvCalc(uplo, hA, N, lda, hXCopy, hXGold, incx);

    double maxRelativeError = helpers::maxRelativeErrorComplexVector(hXGold, hX, N, incx);

    double maxAbsoluteError = helpers::maxAbsoluteErrorComplexVector(hXGold, hX, N, incx);

    std::cout << "max relative err = " << maxRelativeError
              << ", max absolute err = " << maxAbsoluteError << std::endl;

    rstatus = rocblas_destroy_handle(handle);
    CHECK_ROCBLAS_STATUS(rstatus);

    return EXIT_SUCCESS;
}
