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
    helpers::ArgParser options("xn");
    if(!options.validArgs(argc, argv))
        return EXIT_FAILURE;

    //Initialize HIP error to check the return status of the HIP API functions
    hipError_t herror = hipSuccess;

    //Initialize rocBLAS error to check the return status of the rocBLAS API functions
    rocblas_status rstatus = rocblas_status_success;

    //Stride between consecutive values of input vector X (default value is 1)
    rocblas_int incx = options.incx;

    //Number of elements in input vector X and input vector Y (default value is 5)
    rocblas_int n = options.n;

    //Edge condition check
    if(n <= 0)
    {
        std::cout << "Value of 'n' should be greater than 0" << std::endl;
        return 0;
    }

    //Adjusting the size of input vector X for value of stride (incx) not equal to 1
    size_t sizeX = (n * incx) >= 0 ? n * incx : -(n * incx);

    //Allocating memory for the host input vector X and the host scalar result
    std::vector<float> hX(sizeX);
    float              hResult = 0.0;

    //Initialising random values to the host vector X
    helpers::fillVectorNormRand<float>(hX);

    std::cout << "Input Vectors (X)" << std::endl;
    helpers::printVector(hX);

    //accumulate is used to store the sum of squares of vector X
    float accumulate = 0.0;

    /*goldResult is used to store the square root of accumulate
    and is used to compare our result from rocBLAS NRM2 funtion*/
    float goldResult = 0.0;

    //CPU function for NRM2
    for(int i = 0; i < n; i++)
        accumulate += (hX[i * incx] * hX[i * incx]);

    goldResult = sqrt(accumulate);

    //Using rocblas API to create a handle
    rocblas_handle handle;
    rstatus = rocblas_create_handle(&handle);
    CHECK_ROCBLAS_STATUS(rstatus);

    {
        //Allocating memory for the device vector X
        helpers::DeviceVector<float> dX(sizeX);

        //Enable passing hResult parameter from pointer to host memory
        rstatus = rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
        CHECK_ROCBLAS_STATUS(rstatus);

        //Tansfer data from host vector X to device vector X
        herror = hipMemcpy(dX, hX.data(), sizeof(float) * sizeX, hipMemcpyHostToDevice);
        CHECK_HIP_ERROR(herror);

        //Asynchronous NRM2 calculation on device
        rstatus = rocblas_snrm2(handle, n, dX, incx, &hResult);

        CHECK_ROCBLAS_STATUS(rstatus);

        //block until result is ready
        CHECK_HIP_ERROR(hipDeviceSynchronize());

    } // release device memory via helpers::DeviceVector destructors

    //Print GPU generated output
    std::cout << "Output result" << std::endl;
    std::cout << hResult << std::endl;

    //Print the CPU generated output
    std::cout << "Output Goldstandard result" << std::endl;
    std::cout << goldResult << std::endl;

    rstatus = rocblas_destroy_handle(handle);
    CHECK_ROCBLAS_STATUS(rstatus);
    return 0;
}
//End of the program
