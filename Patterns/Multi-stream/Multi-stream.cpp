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
#include <omp.h>
#include <rocblas/rocblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

const int NUM_STREAMS = 4;

int main(int argc, char** argv)
{
    helpers::ArgParser options("Nabxy");
    if(!options.validArgs(argc, argv))
        return EXIT_FAILURE;

    //Initialize rocBLAS error to check the return status of the rocBLAS API functions
    rocblas_status rstatus = rocblas_status_success;

    //Initialize HIP error to check the return status of the HIP API functions
    hipError_t herror = hipSuccess;

    //Number of elements (default value is 5)
    rocblas_int N = options.N;

    //Stride between consecutive values of input vector 'X' (default value is 1)
    rocblas_int incx = options.incx;

    //Stride between consecutive values of input vector 'Y' (default value is 1)
    rocblas_int incy = options.incy;

    //Scalar 'alpha' value used for multiplication
    float h_Alpha = options.alpha;

    //Scalar 'beta' value used for multiplication
    float h_Beta = options.beta;

    //Edge condition check
    if(N <= 0)
    {
        std::cout << "The value of 'n' should be greater than zero" << std::endl;
        return 0;
    }

    //Size of input Matrix 'A'
    rocblas_int lda    = N;
    size_t      size_A = lda * size_t(N) * NUM_STREAMS;

    //Absolute Value of 'incx' and 'incy'
    size_t abs_incx = incx >= 0 ? incx : -incx;
    size_t abs_incy = incy >= 0 ? incy : -incy;

    //Adjusting the size of input vector 'X' for value of stride ('incx') not equal to 1
    size_t size_X = N * abs_incx * NUM_STREAMS;

    //Adjusting the size of input vector 'Y' for value of stride ('incy') not equal to 1
    size_t size_Y = N * abs_incy * NUM_STREAMS;

    //Allocating memory for both the host input matrix 'A' and input vectors 'X', 'Y'
    std::vector<float> h_A(size_A);
    std::vector<float> h_X(size_X);
    std::vector<float> h_Y(size_Y);

    //Creating a Identity matrix 'A' for size 'N * N * NUM_STREAMS'
    for(int stream_Id = 0; stream_Id < NUM_STREAMS; stream_Id++)
    {
        //'host_start_offset_A' points to the starting address to create a identity matrix
        int host_start_offset_A = stream_Id * N * N;

        helpers::matIdentity(h_A.data() + host_start_offset_A, N, N, lda);
    }

    //Intialising random values to both the host vectors 'X' and 'Y'
    helpers::fillVectorNormRand(h_X);
    helpers::fillVectorNormRand(h_Y);

    // print input
    std::cout << "Input host Vector (X)" << std::endl;
    //helpers::printVector(h_X);

    std::cout << "Input host Vector (Y)" << std::endl;
    //helpers::printVector(h_Y);

    /*Initialising the values for vector 'h_Y_Gold', this vector will be used as a Gold Standard
    to compare our results from rocBLAS SYMV funtion*/
    std::vector<float> h_Y_Gold(h_Y);
    h_Y_Gold = h_Y;

    //Matrix is identity so just doing simpler calculation over vectors
    for(int stream_Id = 0; stream_Id < NUM_STREAMS; stream_Id++)
    {
        int start_offset = stream_Id * N;
        for(int i = 0; i < N; i++)
            h_Y_Gold[start_offset + i * abs_incy]
                = h_Alpha * 1.0f * h_X[start_offset + i * abs_incx]
                  + h_Beta * h_Y_Gold[start_offset + i * abs_incy];
    }

    //Allocating different handles for different streams
    rocblas_handle handles[NUM_STREAMS];
    hipStream_t    streams[NUM_STREAMS];

    //Using rocblas API to create handles and hip API to create streams
    for(int stream_Id = 0; stream_Id < NUM_STREAMS; stream_Id++)
    {
        rstatus = rocblas_create_handle(&handles[stream_Id]);
        CHECK_ROCBLAS_STATUS(rstatus);

        herror = hipStreamCreate(&streams[stream_Id]);
        CHECK_HIP_ERROR(herror);
    }

    //rocblas_fill indicates upper triangular mode
    const rocblas_fill uplo = rocblas_fill_upper;

    {
        //Allocating device memory for input Matrix 'A' and for both the input vectors 'X', 'Y'
        helpers::DeviceVector<float> d_A(size_A);
        helpers::DeviceVector<float> d_X(size_X);
        helpers::DeviceVector<float> d_Y(size_Y);

        if((!d_A && size_A) || (!d_X && size_X) || (!d_Y && size_Y))
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return EXIT_FAILURE;
        }

        //Time data to device, computation, and data from device back to host
        helpers::GPUTimer gpuTimer;
        gpuTimer.start();

//Asynchronously queuing up the work in the device (GPU) by  multiple-host (Multi-CPU)
#pragma omp for
        for(int stream_Id = 0; stream_Id < NUM_STREAMS; stream_Id++)
        {
            //Associate each handle with a stream
            rocblas_set_stream(handles[stream_Id], streams[stream_Id]);

            //'start_offset_A' points to the starting address of matrix 'A' from where the data needs to be transferred from host to device
            size_t start_offset_A = size_t(stream_Id) * N * N;

            herror = hipMemcpyAsync(d_A.data() + start_offset_A,
                                    h_A.data() + start_offset_A,
                                    sizeof(float) * N * N,
                                    hipMemcpyHostToDevice,
                                    streams[stream_Id]);
            CHECK_HIP_ERROR(herror);

            //'start_offset_X' points to the starting address of vector 'X' from where the data needs to be transferred from host to device
            size_t start_offset_X = size_t(stream_Id) * N * abs_incx;

            herror = hipMemcpyAsync(d_X.data() + start_offset_X,
                                    h_X.data() + start_offset_X,
                                    sizeof(float) * N * abs_incx,
                                    hipMemcpyHostToDevice,
                                    streams[stream_Id]);
            CHECK_HIP_ERROR(herror);

            //'start_offset_Y' points to the starting address of vector 'Y' from where the data needs to be transferred from host to device
            size_t start_offset_Y = stream_Id * N * abs_incy;
            herror                = hipMemcpyAsync(d_Y.data() + start_offset_Y,
                                    h_Y.data() + start_offset_Y,
                                    sizeof(float) * N * abs_incy,
                                    hipMemcpyHostToDevice,
                                    streams[stream_Id]);
            CHECK_HIP_ERROR(herror);

            //Enable passing alpha parameter from pointer to host memory
            rstatus = rocblas_set_pointer_mode(handles[stream_Id], rocblas_pointer_mode_host);
            CHECK_ROCBLAS_STATUS(rstatus);

            //asynchronous calculation on device, returns before finished calculations
            rstatus = rocblas_ssymv(handles[stream_Id],
                                    uplo,
                                    N,
                                    &h_Alpha,
                                    d_A.data() + start_offset_A,
                                    lda,
                                    d_X.data() + start_offset_X,
                                    (rocblas_int)abs_incx,
                                    &h_Beta,
                                    d_Y.data() + start_offset_Y,
                                    (rocblas_int)abs_incy);

            //check that calculation was launched correctly on device, not that result was computed yet
            CHECK_ROCBLAS_STATUS(rstatus);

            //Asynchronous memory transfer from the device to the host
            herror = hipMemcpyAsync(h_Y.data() + start_offset_Y,
                                    d_Y.data() + start_offset_Y,
                                    sizeof(float) * N * abs_incy,
                                    hipMemcpyDeviceToHost,
                                    streams[stream_Id]);
            CHECK_HIP_ERROR(herror);

            herror = hipStreamSynchronize(streams[stream_Id]);
            CHECK_HIP_ERROR(herror);
        }

        /*//Blocks until all work in the streams are complete.
        herror = hipDeviceSynchronize();
        CHECK_HIP_ERROR(herror);*/

        gpuTimer.stop();
    } // release device memory via helpers::DeviceVector destructors

    //Using rocblas API to destroy handles and hip API to destroy streams
    for(int stream_Id = 0; stream_Id < NUM_STREAMS; stream_Id++)
    {
        rstatus = rocblas_destroy_handle(handles[stream_Id]);
        CHECK_ROCBLAS_STATUS(rstatus);

        herror = hipStreamDestroy(streams[stream_Id]);
        CHECK_HIP_ERROR(herror);
    }

    // print input
    std::cout << "Output generated by GPU" << std::endl;
    helpers::printVector(h_Y);

    std::cout << "Output generated by CPU" << std::endl;
    helpers::printVector(h_Y_Gold);

    float max_relative_error = (float)helpers::maxRelativeError(h_Y, h_Y_Gold);
    float eps                = std::numeric_limits<float>::epsilon();
    float tolerance          = 10;

    if(max_relative_error > eps * tolerance)
    {
        std::cout << "FAIL";
    }
    else
    {
        std::cout << "PASS";
    }

    std::cout << ": max. relative err. = " << max_relative_error << std::endl;
    return EXIT_SUCCESS;
}
//End of program
