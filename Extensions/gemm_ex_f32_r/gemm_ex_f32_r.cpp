/*
Copyright 2019-2020 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "helpers.hpp"
#include <hip/hip_runtime.h>
#include <math.h>
#include <rocblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

int main(int argc, char** argv)
{
    helpers::ArgParser options("MNKab");
    // set defaults
    options.M     = 200;
    options.N     = 60;
    options.K     = 200;
    options.alpha = 1.0f;
    options.beta  = 1.0f;

    if(!options.validArgs(argc, argv))
        return EXIT_FAILURE;

    rocblas_status rstatus = rocblas_status_success;

    rocblas_int M = options.M;
    rocblas_int N = options.N;
    rocblas_int K = options.K;

    constexpr rocblas_datatype aType = rocblas_datatype_f32_r; // _r for real vs. _c for complex
    constexpr rocblas_datatype bType = rocblas_datatype_f32_r;
    constexpr rocblas_datatype cType = rocblas_datatype_f32_r;
    constexpr rocblas_datatype dType = rocblas_datatype_f32_r;
    constexpr rocblas_datatype computeType = rocblas_datatype_f32_r;

    rocblas_gemm_algo algo          = rocblas_gemm_algo_standard;
    int32_t           solutionIndex = 0;
    uint32_t          flags         = 0;

    float hAlpha = options.alpha; // Same datatype as compute_type
    float hBeta  = options.beta; // Same datatype as compute_type

    const rocblas_operation transA = rocblas_operation_transpose;
    const rocblas_operation transB = rocblas_operation_none;

    rocblas_int lda, ldb, ldc, ldd, sizeA, sizeB, sizeC, sizeD;
    int         strideA1, strideA2, strideB1, strideB2;

    if(transA == rocblas_operation_none)
    {
        lda      = M;
        sizeA    = K * lda;
        strideA1 = 1;
        strideA2 = lda;
    }
    else
    {
        lda      = K;
        sizeA    = M * lda;
        strideA1 = lda;
        strideA2 = 1;
    }
    if(transB == rocblas_operation_none)
    {
        ldb      = K;
        sizeB    = N * ldb;
        strideB1 = 1;
        strideB2 = ldb;
    }
    else
    {
        ldb      = N;
        sizeB    = K * ldb;
        strideB1 = ldb;
        strideB2 = 1;
    }
    ldc   = M;
    sizeC = N * ldc;
    ldd   = M;
    sizeD = N * ldd;

    // using rocblas API
    rocblas_handle handle;
    rstatus = rocblas_create_handle(&handle);
    CHECK_ROCBLAS_STATUS(rstatus);

    // Naming: dX is in GPU (device) memory. hX is in CPU (host) memory
    std::vector<float> hA(sizeA);
    std::vector<float> hB(sizeB);
    std::vector<float> hC(sizeC);
    std::vector<float> hD(sizeD);
    std::vector<float> hDGold(sizeD);

    helpers::fillVectorUniformIntRand(hA, 1, 3);
    helpers::fillVectorUniformIntRand(hB, 1, 3);
    helpers::fillVectorUniformIntRand(hC, 1, 3);
    helpers::fillVectorUniformIntRand(hD, 1, 3);

    hDGold = hD;

    {
        // allocate memory on device
        helpers::DeviceVector<float> dA(sizeA);
        helpers::DeviceVector<float> dB(sizeB);
        helpers::DeviceVector<float> dC(sizeC);
        helpers::DeviceVector<float> dD(sizeD);

        if(!dA || !dB || !dC || !dD)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return EXIT_FAILURE;
        }

        // copy data from CPU to device
        CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(float) * sizeA, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(float) * sizeB, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(
            dC, static_cast<void*>(hC.data()), sizeof(float) * sizeC, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(
            dD, static_cast<void*>(hD.data()), sizeof(float) * sizeD, hipMemcpyHostToDevice));

        // enable passing alpha parameter from pointer to host memory
        rstatus = rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
        CHECK_ROCBLAS_STATUS(rstatus);

        // asynchronous calculation on device, returns before finished calculations
        rstatus = rocblas_gemm_ex(handle,
                                  transA,
                                  transB,
                                  M,
                                  N,
                                  K,
                                  &hAlpha,
                                  dA,
                                  aType,
                                  lda,
                                  dB,
                                  bType,
                                  ldb,
                                  &hBeta,
                                  dC,
                                  cType,
                                  ldc,
                                  dD,
                                  dType,
                                  ldd,
                                  computeType,
                                  algo,
                                  solutionIndex,
                                  flags);

        // check that calculation was launched correctly on device, not that result
        // was computed yet
        CHECK_ROCBLAS_STATUS(rstatus);

        // fetch device memory results, automatically blocked until results ready
        CHECK_HIP_ERROR(hipMemcpy(hD.data(), dD, sizeof(float) * sizeD, hipMemcpyDeviceToHost));

    } // release device memory via helpers::DeviceVector destructors

    std::cout << "M, N, K, lda, ldb, ldc, ldd = " << M << ", " << N << ", " << K << ", " << lda
              << ", " << ldb << ", " << ldc << ", " << ldd << std::endl;

    // calculate gold standard using CPU
    helpers::matMatMultMixPrec(hAlpha,
                               hBeta,
                               M,
                               N,
                               K,
                               hA.data(),
                               strideA1,
                               strideA2,
                               hB.data(),
                               strideB1,
                               strideB2,
                               hC.data(),
                               1,
                               ldc,
                               hDGold.data(),
                               1,
                               ldd);

    double maxRelativeError = helpers::maxRelativeError(hD, hDGold);
    double eps              = std::numeric_limits<float>::epsilon();
    double tolerance        = 10;
    if(maxRelativeError > eps * tolerance)
    {
        std::cout << "FAIL";
    }
    else
    {
        std::cout << "PASS";
    }
    std::cout << ": max. relative err. = " << maxRelativeError << std::endl;

    rstatus = rocblas_destroy_handle(handle);
    CHECK_ROCBLAS_STATUS(rstatus);

    return EXIT_SUCCESS;
}
