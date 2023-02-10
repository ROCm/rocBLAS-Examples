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
    helpers::ArgParser options("axyn");
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

    if(n <= 0) //Edge condition check
    {
        std::cout << "Value of 'n' should be greater than 0" << std::endl;
        return 0;
    }

    //Scalar value used for multiplication
    float hAlpha = options.alpha;

    //Adjusting the size of input vector X for value of stride (incx) not equal to 1
    size_t sizeX = (n * incx) >= 0 ? n * incx : -(n * incx);

    //Adjusting the size of input vector Y for value of stride (incy) not equal to 1
    size_t sizeY = (n * incy) >= 0 ? n * incy : -(n * incy);

    //Allocating memory for both the host input vectors X and Y
    std::vector<float> hX(sizeX);
    std::vector<float> hY(sizeY);

    //Intialising random values to both the host vectors X and Y
    helpers::fillVectorNormRand<float>(hX);
    helpers::fillVectorNormRand<float>(hY);

    std::cout << "Input Vectors (X)" << std::endl;
    helpers::printVector(hX);

    std::cout << "Input Vectors (Y)" << std::endl;
    helpers::printVector(hY);

    /*Initialising the values for vector hYGold, this vector will be used as a Gold Standard
    to compare our results from rocBLAS SAXPY funtion*/
    std::vector<float> hYGold(hY);

    //CPU function for SAXPY
    for(int i = 0; i < n; i++)
        hYGold[i * incy] = hAlpha * hX[i * incx] + hY[i * incy];

    //Using rocblas API to create a handle
    rocblas_handle handle;
    rstatus = rocblas_create_handle(&handle);
    CHECK_ROCBLAS_STATUS(rstatus);

    {
        //Allocating memory for both the both device vectors X and Y
        helpers::DeviceVector<float> dX(sizeX);
        helpers::DeviceVector<float> dY(sizeY);

        //Tansfer data from host vector X to device vector X
        herror = hipMemcpy(dX, hX.data(), sizeof(float) * sizeX, hipMemcpyHostToDevice);
        CHECK_HIP_ERROR(herror);

        //Tansfer data from host vector Y to device vector Y
        herror = hipMemcpy(dY, hY.data(), sizeof(float) * sizeY, hipMemcpyHostToDevice);
        CHECK_HIP_ERROR(herror);

        //Enable passing alpha parameter from pointer to host memory
        rstatus = rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
        CHECK_ROCBLAS_STATUS(rstatus);

        //Saxpy calculation on device
        rstatus = rocblas_saxpy(handle, n, &hAlpha, dX, incx, dY, incy);

        CHECK_ROCBLAS_STATUS(rstatus);

        /*Transfer the result from device vector Y to host vector Y,
        automatically blocked until results ready*/
        herror = hipMemcpy(hY.data(), dY, sizeof(float) * sizeY, hipMemcpyDeviceToHost);

        CHECK_HIP_ERROR(herror);
    } // release device memory via helpers::DeviceVector destructors

    std::cout << "Output Vector Y" << std::endl;

    //Print output result Vector
    helpers::printVector(hY);

    //Print the CPU generated output
    std::cout << "Output Vector YGold" << std::endl;
    helpers::printVector(hYGold);

    /*Helper function to check the Relative error between output generated
    from rocBLAS API saxpy and the CPU function*/
    float maxRelativeError = (float)helpers::maxRelativeError(hY, hYGold);
    float eps              = std::numeric_limits<float>::epsilon();
    float tolerance        = 10;

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
    return 0;
}
//End of the program
