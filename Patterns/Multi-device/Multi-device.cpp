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

//Number of streams to be launched per device
const int NUM_STREAMS = 2;

int main(int argc, char** argv)
{
    helpers::ArgParser options("MNKab");
    if(!options.validArgs(argc, argv))
        return EXIT_FAILURE;

    //Initialize rocBLAS error to check the return status of the rocBLAS API functions
    rocblas_status rstatus = rocblas_status_success;

    //Initialize HIP error to check the return status of the HIP API functions
    hipError_t herror = hipSuccess;

    //Determine the number of available GPU devices
    int NUM_DEVICES;
    herror = hipGetDeviceCount(&NUM_DEVICES);
    CHECK_HIP_ERROR(herror);
    std::cout << "The total number of available GPU devices are " << NUM_DEVICES << std::endl;

    /*'M' and 'K' determines the number of rows and columns respectively in Matrix 'A' (no transpose)
      'K' and 'N' determines the number of rows and columns respectively in Matrix 'B' (no transpose)
      'M' and 'N' determines the number of rows and columns respectively in Matrix 'C' (no transpose)*/
    rocblas_int M = options.M;
    rocblas_int N = options.N;
    rocblas_int K = options.K;

    //Scalar 'alpha' value used for multiplication
    float h_Alpha = options.alpha;

    //Scalar 'beta' value used for multiplication
    float h_Beta = options.beta;

    //Edge condition check
    if(K < 0 || M < 0 || N < 0)
    {
        std::cout << "The value of 'N', 'M' and 'K' should be greater than zero" << std::endl;
        return 0;
    }

    //rocblas_operation set to no transpose
    const rocblas_operation transA = rocblas_operation_none;
    const rocblas_operation transB = rocblas_operation_none;

    rocblas_int lda, ldb, ldc, size_A, device_size_A, stream_size_A, size_B, device_size_B,
        stream_size_B, size_C, device_size_C, stream_size_C;
    int strideA1, strideA2, strideB1, strideB2;

    //Initializing matrix dimensions according to the rocblas_operation
    if(transA == rocblas_operation_none)
    {
        lda           = M;
        size_A        = K * lda * NUM_DEVICES * NUM_STREAMS;
        device_size_A = K * lda * NUM_STREAMS;
        stream_size_A = K * lda;
        strideA1      = 1;
        strideA2      = lda;
    }
    else
    {
        lda           = K;
        size_A        = M * lda * NUM_DEVICES * NUM_STREAMS;
        device_size_A = M * lda * NUM_STREAMS;
        stream_size_A = M * lda;
        strideA1      = lda;
        strideA2      = 1;
    }
    if(transB == rocblas_operation_none)
    {
        ldb           = K;
        size_B        = N * ldb * NUM_DEVICES * NUM_STREAMS;
        device_size_B = N * ldb * NUM_STREAMS;
        stream_size_B = N * ldb;
        strideB1      = 1;
        strideB2      = ldb;
    }
    else
    {
        ldb           = N;
        size_B        = K * ldb * NUM_DEVICES * NUM_STREAMS;
        device_size_B = K * ldb * NUM_STREAMS;
        stream_size_B = K * ldb;
        strideB1      = ldb;
        strideB2      = 1;
    }
    ldc           = M;
    size_C        = N * ldc * NUM_DEVICES * NUM_STREAMS;
    device_size_C = N * ldc * NUM_STREAMS;
    stream_size_C = N * ldc;

    /*Allocating memory for the host input matrices 'A', 'B' and 'C'. 'h_Gold' matrix will be used as a Gold Standard
    to compare our results from rocBLAS GEMM funtion*/
    std::vector<float> h_A(size_A);
    std::vector<float> h_B(size_B);
    std::vector<float> h_C(size_C);
    std::vector<float> h_Gold(size_C);

    //Helper functions to initialize the matrices
    helpers::fillVectorNormRand(h_A);
    helpers::fillVectorNormRand(h_B);
    helpers::fillVectorNormRand(h_C);
    h_Gold = h_C;

    //Time data to device, computation, and data from device back to host
    helpers::GPUTimer gpuTimer;
    gpuTimer.start();

//Setting up a device per CPU
#pragma omp parallel for
    for(int device_Id = 0; device_Id < NUM_DEVICES; device_Id++)
    {
        herror = hipSetDevice(device_Id);
        CHECK_HIP_ERROR(herror);

        // Create the built-in struct responsible for containing device properties
        hipDeviceProp_t deviceProp;

        //Call the HIP function to get the properties for the particular device
        herror = hipGetDeviceProperties(&deviceProp, device_Id);
        CHECK_HIP_ERROR(herror);

        //Allocating device memory for input matrices 'A', 'B' and 'C'
        helpers::DeviceVector<float> d_A(device_size_A);
        helpers::DeviceVector<float> d_B(device_size_B);
        helpers::DeviceVector<float> d_C(device_size_C);

        if(!d_A || !d_B || !d_C)
        {
            std::cout << "Insufficient memory in Device " << device_Id
                      << " and Name: " << deviceProp.name << std::endl;
            herror = hipErrorOutOfMemory;
            CHECK_HIP_ERROR(herror);
        }

        //Allocating different handles for different streams per device
        rocblas_handle handles[NUM_STREAMS];
        hipStream_t    streams[NUM_STREAMS];

        //Using rocblas API to create handles and hip API to create streams
        for(rocblas_int i = 0; i < NUM_STREAMS; i++)
        {
            rstatus = rocblas_create_handle(&handles[i]);
            CHECK_ROCBLAS_STATUS(rstatus);

            herror = hipStreamCreate(&streams[i]);
            CHECK_HIP_ERROR(herror);
        }

        //Using multiple streams to distribute the data in a particular device
        for(int stream_Id = 0; stream_Id < NUM_STREAMS; stream_Id++)
        {
            //Associate each handle with a stream
            rocblas_set_stream(handles[stream_Id], streams[stream_Id]);

            //'host_start_offset_A' points to the starting address of the host matrix 'A' from where the data needs to be transferred from the host to the device
            size_t host_start_offset_A
                = device_Id > 0 ? device_Id * NUM_STREAMS * M * K : stream_Id * M * K;
            host_start_offset_A = stream_Id > 0 && device_Id > 0 ? host_start_offset_A * stream_Id
                                                                 : host_start_offset_A;

            //'device_start_offset_A' points to the starting address of the device matrix 'A' from where the data needs to be transferred from the host to the device
            size_t device_start_offset_A = stream_Id * M * K;

            herror = hipMemcpyAsync(d_A.data() + device_start_offset_A,
                                    h_A.data() + host_start_offset_A,
                                    sizeof(float) * stream_size_A,
                                    hipMemcpyHostToDevice,
                                    streams[stream_Id]);
            CHECK_HIP_ERROR(herror);

            //'host_start_offset_B' points to the starting address of host matrix 'B' from where the data needs to be transferred from the host to the device
            size_t host_start_offset_B
                = device_Id > 0 ? device_Id * NUM_STREAMS * K * N : stream_Id * K * N;
            host_start_offset_B = stream_Id > 0 && device_Id > 0 ? host_start_offset_B * stream_Id
                                                                 : host_start_offset_B;

            //'device_start_offset_B' points to the starting address of the device matrix 'B' from where the data needs to be transferred from the host to the device
            int device_start_offset_B = stream_Id * K * N;
            herror                    = hipMemcpyAsync(d_B.data() + device_start_offset_B,
                                    h_B.data() + host_start_offset_B,
                                    sizeof(float) * stream_size_B,
                                    hipMemcpyHostToDevice,
                                    streams[stream_Id]);
            CHECK_HIP_ERROR(herror);

            //'host_start_offset_C' points to the starting address of host matrix 'C' from where the data needs to be transferred from the host to the device
            size_t host_start_offset_C
                = device_Id > 0 ? device_Id * NUM_STREAMS * M * N : stream_Id * M * N;
            host_start_offset_C = stream_Id > 0 && device_Id > 0 ? host_start_offset_C * stream_Id
                                                                 : host_start_offset_C;

            //'device_start_offset_C' points to the starting address of the device matrix 'C' from where the data needs to be transferred from the host to the device
            int device_start_offset_C = stream_Id * M * N;
            herror                    = hipMemcpyAsync(d_C.data() + device_start_offset_C,
                                    h_C.data() + host_start_offset_C,
                                    sizeof(float) * stream_size_C,
                                    hipMemcpyHostToDevice,
                                    streams[stream_Id]);
            CHECK_HIP_ERROR(herror);

            //Enable passing alpha parameter from pointer to host memory
            rstatus = rocblas_set_pointer_mode(handles[stream_Id], rocblas_pointer_mode_host);
            CHECK_ROCBLAS_STATUS(rstatus);

            //Asynchronous calculation on device, returns before finished calculations
            rstatus = rocblas_sgemm(handles[stream_Id],
                                    transA,
                                    transB,
                                    M,
                                    N,
                                    K,
                                    &h_Alpha,
                                    d_A.data() + device_start_offset_A,
                                    lda,
                                    d_B.data() + device_start_offset_B,
                                    ldb,
                                    &h_Beta,
                                    d_C.data() + device_start_offset_C,
                                    ldc);

            //Check that calculation was launched correctly on device, not that result was computed yet
            CHECK_ROCBLAS_STATUS(rstatus);

            //Fetch device memory results
            herror = hipMemcpyAsync(h_C.data() + host_start_offset_C,
                                    d_C.data() + device_start_offset_C,
                                    sizeof(float) * stream_size_C,
                                    hipMemcpyDeviceToHost,
                                    streams[stream_Id]);
            CHECK_HIP_ERROR(herror);
        }

        //Blocks until all work in the streams are complete.
        herror = hipDeviceSynchronize();
        CHECK_HIP_ERROR(herror);

        //Using rocblas API to destroy handles and hip API to destroy streams
        for(rocblas_int i = 0; i < NUM_STREAMS; i++)
        {
            rstatus = rocblas_destroy_handle(handles[i]);
            CHECK_ROCBLAS_STATUS(rstatus);

            herror = hipStreamDestroy(streams[i]);
            CHECK_HIP_ERROR(herror);
        }

    } // release device memory via helpers::DeviceVector destructors

    gpuTimer.stop();

    //Calculate gold standard result using CPU
    for(int device_Id = 0; device_Id < NUM_DEVICES; device_Id++)
    {
        for(int stream_Id = 0; stream_Id < NUM_STREAMS; stream_Id++)
        {
            //'start_offset_A' points to the starting address of matrix 'A' which is needed for 'matMatMult' helper function
            int start_offset_A
                = device_Id > 0 ? device_Id * NUM_STREAMS * M * K : stream_Id * M * K;
            start_offset_A
                = stream_Id > 0 && device_Id > 0 ? start_offset_A * stream_Id : start_offset_A;

            //'start_offset_B' points to the starting address of matrix 'B' which is needed for 'matMatMult' helper function
            int start_offset_B
                = device_Id > 0 ? device_Id * NUM_STREAMS * K * N : stream_Id * K * N;
            start_offset_B
                = stream_Id > 0 && device_Id > 0 ? start_offset_B * stream_Id : start_offset_B;

            //'start_offset_C' points to the starting address of matrix 'C' which is needed for 'matMatMult' helper function
            int start_offset_C
                = device_Id > 0 ? device_Id * NUM_STREAMS * M * N : stream_Id * M * N;
            start_offset_C
                = stream_Id > 0 && device_Id > 0 ? start_offset_C * stream_Id : start_offset_C;

            helpers::matMatMult<float>(h_Alpha,
                                       h_Beta,
                                       M,
                                       N,
                                       K,
                                       h_A.data() + start_offset_A,
                                       strideA1,
                                       strideA2,
                                       h_B.data() + start_offset_B,
                                       strideB1,
                                       strideB2,
                                       h_Gold.data() + start_offset_C,
                                       1,
                                       ldc);
        }
    }

    //std::cout << "Output Matrix generated by GPU (X)" << std::endl;
    //helpers::printVector(h_C);

    //std::cout << "Output Matrix generated by CPU (X)" << std::endl;
    //helpers::printVector(h_Gold);

    std::cout << "M, N, K, lda, ldb, ldc, NUM_STREAMS, NUM_DEVICES = " << M << ", " << N << ", "
              << K << "," << lda << ", " << ldb << ", " << ldc << ", " << NUM_STREAMS << ", "
              << NUM_DEVICES << std::endl;

    float max_error     = (float)helpers::maxError(h_C, h_Gold);
    float eps           = std::numeric_limits<float>::epsilon();
    float tolerance     = 10; // if tests fail try increasing tolerance
    float allowed_error = eps * tolerance * K * sqrtf((float)K); // allows for roundoff error

    if(max_error > allowed_error)
    {
        std::cout << "FAIL";
    }
    else
    {
        std::cout << "PASS";
    }

    std::cout << ": max_err = " << max_error << std::endl;

    return EXIT_SUCCESS;
}
//End of program
