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

    helpers::ArgParser options("nxy");
    if(!options.validArgs(argc, argv))
        return EXIT_FAILURE;

    hipError_t     herror  = hipSuccess;
    rocblas_status rstatus = rocblas_status_success;

    rocblas_int incx = options.incx;
    rocblas_int incy = options.incy;
    rocblas_int n    = options.n;

    // enlarge nSize to allow for input parameters increment
    rocblas_int nSize = n * std::max(incx, incy);

    // host input vectors of size nSize
    std::vector<float> hostVecA(nSize);
    // test data is just random numbers set by helper functions
    helpers::fillVectorNormRand(hostVecA, incx);
    std::vector<float> hostVecB(nSize);
    helpers::fillVectorNormRand(hostVecB, incy);

    // print input
    std::cout << "Input Vectors" << std::endl;
    helpers::printVector(hostVecA, n, incx);
    helpers::printVector(hostVecB, n, incy);

    size_t vectorBytes = nSize * sizeof(float);

    // allocate device vector memory using hipMalloc and copy data from host
    float* deviceVecA;
    herror = hipMalloc(&deviceVecA, vectorBytes);
    CHECK_HIP_ERROR(herror);
    rstatus = rocblas_set_vector(nSize, sizeof(float), hostVecA.data(), incx, deviceVecA, incx);
    CHECK_ROCBLAS_STATUS(rstatus);
    // equivalent if increments are all 1 to doing call
    //herror = hipMemcpy(deviceVecA, hostVecA.data(), vectorBytes, hipMemcpyHostToDevice);
    //CHECK_HIP_ERROR(herror);

    float* deviceVecB;
    herror = hipMalloc(&deviceVecB, vectorBytes);
    CHECK_HIP_ERROR(herror);
    rstatus = rocblas_set_vector(nSize, sizeof(float), hostVecB.data(), incy, deviceVecB, incy);
    CHECK_ROCBLAS_STATUS(rstatus);

    // using rocblas API
    rocblas_handle handle;
    rstatus = rocblas_create_handle(&handle);
    CHECK_ROCBLAS_STATUS(rstatus);

    // asynchronous calculation on device, returns before finished calculations
    // Leading 's' in sswap stands for single precision float
    // the rocblas "C" API specifies data type this way as there is no function overloading in "C"
    rstatus = rocblas_sswap(handle, n, deviceVecA, incx, deviceVecB, incy);
    // check that calculation was launched correctly on device, not that result
    // was computed yet
    CHECK_ROCBLAS_STATUS(rstatus);

    // fetch device memory results, hipMemcpy automatically blocked until results ready
    rstatus = rocblas_get_vector(nSize, sizeof(float), deviceVecA, incx, hostVecA.data(), incx);
    CHECK_ROCBLAS_STATUS(rstatus);
    // equivalent if increments are all 1 to doing call
    // herror = hipMemcpy(hostVecA.data(), deviceVecA, vectorBytes, hipMemcpyDeviceToHost);
    // CHECK_HIP_ERROR(herror);

    rstatus = rocblas_get_vector(nSize, sizeof(float), deviceVecB, incy, hostVecB.data(), incy);
    CHECK_ROCBLAS_STATUS(rstatus);

    // print results
    std::cout << "Output Vectors" << std::endl;
    helpers::printVector(hostVecA, n, incx);
    helpers::printVector(hostVecB, n, incy);

    // release device memory
    herror = hipFree(deviceVecA);
    CHECK_HIP_ERROR(herror);
    herror = hipFree(deviceVecB);
    CHECK_HIP_ERROR(herror);

    // releasing rocblas resources
    rstatus = rocblas_destroy_handle(handle);
    CHECK_ROCBLAS_STATUS(rstatus);

    return EXIT_SUCCESS;
}
