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
    helpers::ArgParser options("MNKab");
    if(!options.validArgs(argc, argv))
        return EXIT_FAILURE;

    rocblas_status rstatus = rocblas_status_success;

    typedef float dataType;

    rocblas_int M = options.M;
    rocblas_int N = options.N;
    rocblas_int K = options.K;

    float hAlpha = options.alpha;
    float hBeta  = options.beta;

    const rocblas_operation transA = rocblas_operation_none;
    const rocblas_operation transB = rocblas_operation_none;

    rocblas_int lda, ldb, ldc, sizeA, sizeB, sizeC;
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

    // using rocblas API
    rocblas_handle handle;
    rstatus = rocblas_create_handle(&handle);
    CHECK_ROCBLAS_STATUS(rstatus);

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory

    std::vector<dataType> hA(sizeA, 1);
    std::vector<dataType> hB(sizeB);
    std::vector<dataType> hC(sizeC, 1);
    std::vector<dataType> hGold(sizeC);

    // helpers::matIdentity(hA.data(), M, K, lda);
    helpers::matIdentity(hB.data(), K, N, ldb);
    // helpers::matIdentity(hC.data(), M, N, ldc);
    hGold = hC;

    {
        // allocate memory on device
        helpers::DeviceVector<dataType> dA(sizeA);
        helpers::DeviceVector<dataType> dB(sizeB);
        helpers::DeviceVector<dataType> dC(sizeC);

        if(!dA || !dB || !dC)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return EXIT_FAILURE;
        }

        // copy data from CPU to device
        CHECK_HIP_ERROR(hipMemcpy(
            dA, static_cast<void*>(hA.data()), sizeof(dataType) * sizeA, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(
            dB, static_cast<void*>(hB.data()), sizeof(dataType) * sizeB, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(
            dC, static_cast<void*>(hC.data()), sizeof(dataType) * sizeC, hipMemcpyHostToDevice));

        // enable passing alpha parameter from pointer to host memory
        rstatus = rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
        CHECK_ROCBLAS_STATUS(rstatus);

        // asynchronous calculation on device, returns before finished calculations
        rstatus = rocblas_sgemm(
            handle, transA, transB, M, N, K, &hAlpha, dA, lda, dB, ldb, &hBeta, dC, ldc);

        // check that calculation was launched correctly on device, not that result
        // was computed yet
        CHECK_ROCBLAS_STATUS(rstatus);

        // fetch device memory results, automatically blocked until results ready
        CHECK_HIP_ERROR(hipMemcpy(hC.data(), dC, sizeof(dataType) * sizeC, hipMemcpyDeviceToHost));

    } // release device memory via helpers::DeviceVector destructors

    std::cout << "M, N, K, lda, ldb, ldc = " << M << ", " << N << ", " << K << ", " << lda << ", "
              << ldb << ", " << ldc << std::endl;

    // calculate gold standard using CPU
    helpers::matMatMult<dataType>(hAlpha,
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
                                  hGold.data(),
                                  1,
                                  ldc);

    dataType maxRelativeError = (dataType)helpers::maxRelativeError(hC, hGold);
    dataType eps              = std::numeric_limits<dataType>::epsilon();
    dataType tolerance        = 10;
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
