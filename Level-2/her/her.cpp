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
#include <complex>
#include <hip/hip_complex.h>
#include <hip/hip_runtime_api.h>
#include <math.h>
#include <rocblas/rocblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

int main(int argc, char** argv)
{
    helpers::ArgParser options("Nax");
    if(!options.validArgs(argc, argv))
        return EXIT_FAILURE;

    rocblas_status rstatus = rocblas_status_success;

    rocblas_int N    = options.N;
    rocblas_int incx = options.incx;

    float hAlpha = options.alpha;

    const rocblas_fill uplo = rocblas_fill_upper;

    size_t sizeX, absIncx;

    rocblas_int lda   = N;
    size_t      sizeA = lda * size_t(N);

    absIncx = incx >= 0 ? incx : -incx;

    sizeX = N * absIncx;

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory

    std::vector<hipFloatComplex> hA(sizeA);

    // we are using std::complex for it's operators and it has same memory layout
    // as hipFloatComplex so can copy the data into the array for use in the rocblas C API
    std::vector<std::complex<float>> hX(sizeX);
    helpers::fillVectorUniformIntRand(hX);

    std::vector<hipFloatComplex> hAGold(sizeA);

    // initialize simple data for simple host side reference computation
    helpers::matIdentity(hA.data(), N, N, lda);
    hAGold = hA;

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
        CHECK_HIP_ERROR(
            hipMemcpy(dA, hA.data(), sizeof(hipFloatComplex) * sizeA, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(
            hipMemcpy(dX, hX.data(), sizeof(std::complex<float>) * sizeX, hipMemcpyHostToDevice));

        // enable passing alpha and beta parameters from pointer to host memory
        rstatus = rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
        CHECK_ROCBLAS_STATUS(rstatus);

        // asynchronous calculation on device, returns before finished calculations
        rstatus = rocblas_cher(handle, uplo, N, &hAlpha, dX, incx, dA, lda);

        // check that calculation was launched correctly on device, not that result
        // was computed yet
        CHECK_ROCBLAS_STATUS(rstatus);

        // fetch device memory results, automatically blocked until results ready
        CHECK_HIP_ERROR(
            hipMemcpy(hA.data(), dA, sizeof(hipFloatComplex) * sizeA, hipMemcpyDeviceToHost));

        gpuTimer.stop();

    } // release device memory via helpers::DeviceVector destructors

    std::cout << "alpha, N, lda = " << hAlpha << ", " << N << ", " << lda << std::endl;

    // calculate expected result using CPU
    for(int i = 0; i < N; i++)
    {
        // matrix is identity so just doing simpler calculation over x vectors
        for(int j = 0; j < N; j++)
        {
            std::complex<float> r = hX[j] * std::conj(hX[i]);
            r *= std::complex<float>(hAlpha, 0);

            // using hip helper function hipCaddf to add hip complex type
            hAGold[i * lda + j]
                = hipCaddf(hipFloatComplex(r.real(), r.imag()), hAGold[i * lda + j]);
        }
    }

    bool fail = false;
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            if(uplo == rocblas_fill_upper && j > i)
                continue;
            else if(uplo != rocblas_fill_upper && j < i)
                continue;

            if(hAGold[i * lda + j] != hA[i * lda + j])
                fail = true;
        }
    }

    if(fail)
    {
        std::cout << "FAIL";
    }
    else
    {
        std::cout << "PASS";
    }

    rstatus = rocblas_destroy_handle(handle);
    CHECK_ROCBLAS_STATUS(rstatus);

    return EXIT_SUCCESS;
}
