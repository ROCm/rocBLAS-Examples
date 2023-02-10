/* ************************************************************************
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <hip/hip_runtime_api.h>
#include <math.h>
#include <rocblas/rocblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

int main(int argc, char** argv)
{
    helpers::ArgParser options("MNKabc");
    if(!options.validArgs(argc, argv))
        return EXIT_FAILURE;

    rocblas_status rstatus = rocblas_status_success;

    typedef float dataType;

    rocblas_int M = options.M;
    rocblas_int N = options.N;
    rocblas_int K = options.K;

    float hAlpha = options.alpha;
    float hBeta  = options.beta;

    rocblas_int batchCount = options.batchCount;

    const rocblas_operation transA = rocblas_operation_none;
    const rocblas_operation transB = rocblas_operation_none;

    rocblas_int lda, ldb, ldc; // leading dimension of matrices
    rocblas_int strideA1, strideA2, strideB1, strideB2;

    rocblas_stride strideA, strideB, strideC;

    if(transA == rocblas_operation_none)
    {
        lda      = M;
        strideA  = rocblas_stride(K) * lda;
        strideA1 = 1;
        strideA2 = lda;
    }
    else
    {
        lda      = K;
        strideA  = rocblas_stride(M) * lda;
        strideA1 = lda;
        strideA2 = 1;
    }
    if(transB == rocblas_operation_none)
    {
        ldb      = K;
        strideB  = rocblas_stride(N) * ldb;
        strideB1 = 1;
        strideB2 = ldb;
    }
    else
    {
        ldb      = N;
        strideB  = rocblas_stride(K) * ldb;
        strideB1 = ldb;
        strideB2 = 1;
    }
    ldc     = M;
    strideC = rocblas_stride(N) * ldc;

    rocblas_int cnt        = std::max(batchCount, 1);
    size_t      totalSizeA = size_t(strideA) * cnt;
    size_t      totalSizeB = size_t(strideB) * cnt;
    size_t      totalSizeC = size_t(strideC) * cnt;

    // using rocblas API
    rocblas_handle handle;
    rstatus = rocblas_create_handle(&handle);
    CHECK_ROCBLAS_STATUS(rstatus);

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory

    // using single block of contiguous data for all batches
    std::vector<dataType> hA(totalSizeA, 1);
    std::vector<dataType> hB(totalSizeB);
    std::vector<dataType> hC(totalSizeC, 1);
    std::vector<dataType> hGold(totalSizeC);

    for(int i = 0; i < batchCount; i++)
    {
        helpers::matIdentity(hB.data() + i * strideB, K, N, ldb);
    }
    hGold = hC;

    {
        // allocate memory on device
        helpers::DeviceVector<dataType> dA(totalSizeA);
        helpers::DeviceVector<dataType> dB(totalSizeB);
        helpers::DeviceVector<dataType> dC(totalSizeC);

        if(!dA || !dB || !dC)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return EXIT_FAILURE;
        }

        // copy data from CPU to device
        CHECK_HIP_ERROR(hipMemcpy(dA,
                                  static_cast<void*>(hA.data()),
                                  sizeof(dataType) * totalSizeA,
                                  hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dB,
                                  static_cast<void*>(hB.data()),
                                  sizeof(dataType) * totalSizeB,
                                  hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dC,
                                  static_cast<void*>(hC.data()),
                                  sizeof(dataType) * totalSizeC,
                                  hipMemcpyHostToDevice));

        // enable passing alpha parameter from pointer to host memory
        rstatus = rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
        CHECK_ROCBLAS_STATUS(rstatus);

        // asynchronous calculation on device, returns before finished calculations
        rstatus = rocblas_sgemm_strided_batched(handle,
                                                transA,
                                                transB,
                                                M,
                                                N,
                                                K,
                                                &hAlpha,
                                                dA,
                                                lda,
                                                strideA,
                                                dB,
                                                ldb,
                                                strideB,
                                                &hBeta,
                                                dC,
                                                ldc,
                                                strideC,
                                                batchCount);

        // check that calculation was launched correctly on device, not that result
        // was computed yet
        CHECK_ROCBLAS_STATUS(rstatus);

        // fetch device memory results, automatically blocked until results ready
        CHECK_HIP_ERROR(
            hipMemcpy(hC.data(), dC, sizeof(dataType) * totalSizeC, hipMemcpyDeviceToHost));

    } // release device memory via helpers::DeviceVector destructors

    std::cout << "M, N, K, lda, ldb, ldc = " << M << ", " << N << ", " << K << ", " << lda << ", "
              << ldb << ", " << ldc << std::endl;

    // calculate gold standard using CPU
    for(int i = 0; i < batchCount; i++)
    {
        float* aPtr = &hA[i * strideA];
        float* bPtr = &hB[i * strideB];
        float* cPtr = &hGold[i * strideC];

        helpers::matMatMult<dataType>(hAlpha,
                                      hBeta,
                                      M,
                                      N,
                                      K,
                                      aPtr,
                                      strideA1,
                                      strideA2,
                                      bPtr,
                                      strideB1,
                                      strideB2,
                                      cPtr,
                                      1,
                                      ldc);
    }

    dataType maxRelativeError = (dataType)helpers::maxRelativeError(hC, hGold);
    dataType eps              = std::numeric_limits<dataType>::epsilon();
    float    tolerance        = 10;
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
