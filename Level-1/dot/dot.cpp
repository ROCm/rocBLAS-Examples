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
    helpers::ArgParser options("xyn");
    if(!options.validArgs(argc, argv))
        return EXIT_FAILURE;

    //Initialize HIP error to check the return status of the HIP API functions
    hipError_t herror = hipSuccess;

    //Initialize rocBLAS error to check the return status of the rocBLAS API functions
    rocblas_status rstatus = rocblas_status_success;

    //Stride between consecutive values of input vector X (default value is 1)
    rocblas_int incx = options.incx;

    //Stride between consecutive values of input vector Y (default value is 1)
    rocblas_int incy = options.incy;

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

    //Adjusting the size of input vector Y for value of stride (incy) not equal to 1
    size_t sizeY = (n * incy) >= 0 ? n * incy : -(n * incy);

    //Allocating memory for the host input vectors X, Y and the host scalar result
    std::vector<float> hX(sizeX);
    std::vector<float> hY(sizeY);

    float hResult = 0.0;

    //Initialising random values to both the host vectors X and Y
    helpers::fillVectorNormRand<float>(hX);
    helpers::fillVectorNormRand<float>(hY);

    std::cout << "Input Vectors (X)" << std::endl;
    helpers::printVector(hX);

    std::cout << "Input Vectors (Y)" << std::endl;
    helpers::printVector(hY);

    /*Initialising the scalar goldResult, goldResult will be used as a
    gold standard to compare our result from rocBLAS SDOT funtion*/
    float goldResult = 0.0;

    //CPU function for SDOT
    for(int i = 0; i < n; i++)
        goldResult += hX[i * incx] * hY[i * incy];

    //Using rocblas API to create a handle
    rocblas_handle handle;
    rstatus = rocblas_create_handle(&handle);
    CHECK_ROCBLAS_STATUS(rstatus);

    {
        //Allocating memory for the device vectors X, Y and the scalar Result
        helpers::DeviceVector<float> dX(sizeX);
        helpers::DeviceVector<float> dY(sizeY);

        //Enable passing hResult parameter from pointer to host memory
        rstatus = rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
        CHECK_ROCBLAS_STATUS(rstatus);

        //Tansfer data from host vector X to device vector X
        herror = hipMemcpy(dX, hX.data(), sizeof(float) * sizeX, hipMemcpyHostToDevice);
        CHECK_HIP_ERROR(herror);

        //Tansfer data from host vector Y to device vector Y
        herror = hipMemcpy(dY, hY.data(), sizeof(float) * sizeY, hipMemcpyHostToDevice);
        CHECK_HIP_ERROR(herror);

        //Asynchronous SDOT calculation on device
        rstatus = rocblas_sdot(handle, n, dX, incx, dY, incy, &hResult);

        CHECK_ROCBLAS_STATUS(rstatus);

        //Block until result is ready
        CHECK_HIP_ERROR(hipDeviceSynchronize());

    } // release device memory via helpers::DeviceVector destructors

    //Print the GPU generated output
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
