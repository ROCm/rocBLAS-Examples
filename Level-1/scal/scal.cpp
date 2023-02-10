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

    helpers::ArgParser options("nax");
    if(!options.validArgs(argc, argv))
        return EXIT_FAILURE;

    hipError_t     herror  = hipSuccess;
    rocblas_status rstatus = rocblas_status_success;

    rocblas_int n    = options.n;
    rocblas_int incx = options.incx;
    size_t      size = (n * incx) > 0 ? (n * incx) : -(n * incx);

    typedef double dataType;

    // host input vectors of size n
    std::vector<dataType> hostVecA(size);
    helpers::fillVectorNormRand<dataType>(hostVecA);

    // print input
    std::cout << "Input Vector" << std::endl;
    helpers::printVector<dataType>(hostVecA);

    size_t vectorBytes = size * sizeof(dataType);

    // allocate device vectors and copy memory from host
    dataType* deviceVecA;
    herror = hipMalloc(&deviceVecA, vectorBytes);
    CHECK_HIP_ERROR(herror);
    herror = hipMemcpy(deviceVecA, hostVecA.data(), vectorBytes, hipMemcpyHostToDevice);
    CHECK_HIP_ERROR(herror);

    // using rocblas API
    rocblas_handle handle;
    rstatus = rocblas_create_handle(&handle);
    CHECK_ROCBLAS_STATUS(rstatus);

    double alpha = options.alpha;

    // enable passing alpha parameter from pointer to host memory
    rstatus = rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    CHECK_ROCBLAS_STATUS(rstatus);

    // asynchronous calculation on device, returns before finished calculations
    rstatus = rocblas_dscal(handle, n, &alpha, deviceVecA, incx);
    // check that calculation was launched correctly on device, not that result
    // was computed yet
    CHECK_ROCBLAS_STATUS(rstatus);

    // fetch device memory results, automatically blocked until results ready
    herror = hipMemcpy(hostVecA.data(), deviceVecA, vectorBytes, hipMemcpyDeviceToHost);
    CHECK_HIP_ERROR(herror);

    // print results
    std::cout << "Output Vector, alpha = " << alpha << std::endl;
    helpers::printVector<dataType>(hostVecA);

    // release device memory
    herror = hipFree(deviceVecA);
    CHECK_HIP_ERROR(herror);

    rstatus = rocblas_destroy_handle(handle);
    CHECK_ROCBLAS_STATUS(rstatus);

    return EXIT_SUCCESS;
}
