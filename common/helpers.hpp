/*
Copyright 2019-2020 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once
#include "ArgParser.hpp"
#include "error_macros.h"
#include "memoryHelpers.hpp"
#include "timers.hpp"
#include <complex>
#include <cstdio>
#include <hip/hip_complex.h>
#include <iostream>
#include <random>
#include <rocblas.h>
#include <stdlib.h>
#include <vector>

namespace helpers
{

    template <typename T>
    void printVector(const std::vector<T>& v, size_t n = 0, rocblas_int inc = 1)
    {
        if(n <= 0)
            n = v.size();
        for(size_t i = 0; i < n; i += inc)
        {
            std::cout << v[i] << " ";
        }
        std::cout << "\n";
    }

    template <typename T>
    void printMatrix(const char* name, T* A, rocblas_int m, rocblas_int n, rocblas_int lda)
    {
        printf("---------- %s ----------\n", name);
        for(int i = 0; i < m; i++)
        {
            for(int j = 0; j < n; j++)
            {
                printf("%f ", A[i + j * lda]);
            }
            printf("\n");
        }
    }

    template <typename T>
    void fillVectorNormRand(std::vector<T>& arr, rocblas_int inc = 1)
    {
        // Initialize the array with a normal distributed random variable.

        srand(int(time(NULL)));
        std::random_device          rd{};
        std::mt19937                gen{rd()};
        std::normal_distribution<T> distrib{T(0), T(1)};
        (void) distrib(gen); // prime generator to remove warning

        for(size_t i = 0; i < arr.size(); i += inc)
        {
            arr[i] = distrib(gen);
        }
    }

    /*! \brief  generate a random number in [-0.5,0.5] */
    template <typename T>
    inline T randomHPLGenerator()
    {
        std::random_device rd{};
        std::mt19937       gen{rd()};
        return std::uniform_real_distribution<T>(-0.5, 0.5);
    }

    // Initialize vector with HPL-like random values
    template <typename T>
    inline void randomInitHPL(
        std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batchCount = 1)
    {
        for(size_t iBatch = 0; iBatch < batchCount; iBatch++)
            for(size_t i = 0; i < M; ++i)
                for(size_t j = 0; j < N; ++j)
                    A[i + j * lda + iBatch * stride] = randomHPLGenerator<T>();
    }

    template <typename T,
              std::enable_if_t<!std::is_same<T, std::complex<float>>{}
                                   && !std::is_same<T, hipFloatComplex>{},
                               int> = 0>
    void fillVectorUniformIntRand(std::vector<T>& arr, rocblas_int inc = 1, int range = 3)
    {
        srand(int(time(NULL)));
        std::random_device                 rd{};
        std::mt19937                       gen{rd()};
        std::uniform_int_distribution<int> distrib{-range, range};
        (void) distrib(gen); // prime generator to remove warning

        for(size_t i = 0; i < arr.size(); i += inc)
        {
            int rval = distrib(gen);
            arr[i]   = T((float)rval);
        }
    }

    template <typename T,
              std::enable_if_t<std::is_same<T, std::complex<float>>{}
                                   || std::is_same<T, hipFloatComplex>{},
                               int> = 0>
    void fillVectorUniformIntRand(std::vector<T>& arr, rocblas_int inc = 1, int range = 3)
    {
        srand(int(time(NULL)));
        std::random_device                 rd{};
        std::mt19937                       gen{rd()};
        std::uniform_int_distribution<int> distrib{-range, range};
        distrib(gen); // prime generator to remove warning

        for(size_t i = 0; i < arr.size(); i += inc)
        {
            int rval = distrib(gen);
            int ival = distrib(gen);
            arr[i]   = T(rval, ival);
        }
    }

    template <typename T>
    double maxError(std::vector<T>& A, std::vector<T>& reference)
    {
        double maxError = double(std::numeric_limits<T>::min());

        size_t n = A.size();
        for(size_t i = 0; i < n; ++i)
        {
            double gold  = double(reference[i]);
            double Error = gold != 0 ? gold - double(A[i]) : double(A[i]);
            Error        = Error > 0 ? Error : -Error;
            maxError     = Error < maxError ? maxError : Error;
        }
        return maxError;
    }

    template <typename T>
    double maxRelativeError(std::vector<T>& A, std::vector<T>& reference)
    {
        double maxRelativeError = double(std::numeric_limits<T>::min());

        size_t n = A.size();
        for(size_t i = 0; i < n; ++i)
        {
            double gold          = double(reference[i]);
            double relativeError = gold != 0 ? (gold - double(A[i])) / (gold) : double(A[i]);
            relativeError        = relativeError > 0 ? relativeError : -relativeError;
            maxRelativeError = relativeError < maxRelativeError ? maxRelativeError : relativeError;
        }
        return maxRelativeError;
    }

    template <typename T>
    T matMaxRelativeError(T*     A,
                          T*     reference,
                          size_t M,
                          size_t N,
                          size_t lda,
                          size_t stride     = 0,
                          size_t batchCount = 1)
    {
        T maxRelativeError = std::numeric_limits<T>::min();

        for(size_t iBatch = 0; iBatch < batchCount; iBatch++)
            for(size_t i = 0; i < M; ++i)
                for(size_t j = 0; j < N; ++j)
                {
                    T     gold          = reference[i + j * lda + iBatch * stride];
                    float relativeError = std::numeric_limits<T>::min();
                    if(gold != 0)
                    {
                        relativeError = (gold - A[i + j * lda + iBatch * stride]) / gold;
                    }
                    else
                    {
                        relativeError = A[i + j * lda + iBatch * stride];
                    }
                    relativeError = relativeError > 0 ? relativeError : -relativeError;
                    maxRelativeError
                        = relativeError < maxRelativeError ? maxRelativeError : relativeError;
                }
        return maxRelativeError;
    }

    template <typename T>
    void matMatMult(T   alpha,
                    T   beta,
                    int M,
                    int N,
                    int K,
                    T*  A,
                    int As1,
                    int As2,
                    T*  B,
                    int Bs1,
                    int Bs2,
                    T*  C,
                    int Cs1,
                    int Cs2)
    {
        for(int i1 = 0; i1 < M; i1++)
        {
            for(int i2 = 0; i2 < N; i2++)
            {
                T t = T(0.0);
                for(int i3 = 0; i3 < K; i3++)
                {
                    t += A[i1 * As1 + i3 * As2] * B[i3 * Bs1 + i2 * Bs2];
                }
                C[i1 * Cs1 + i2 * Cs2] = beta * C[i1 * Cs1 + i2 * Cs2] + alpha * t;
            }
        }
    }

    template <typename T, typename U = T>
    void matMatMult(T   alpha,
                    T   beta,
                    int M,
                    int N,
                    int K,
                    U*  A,
                    int As1,
                    int As2,
                    U*  B,
                    int Bs1,
                    int Bs2,
                    T*  C,
                    int Cs1,
                    int Cs2,
                    T*  D,
                    int Ds1,
                    int Ds2)
    {
        for(int i1 = 0; i1 < M; i1++)
        {
            for(int i2 = 0; i2 < N; i2++)
            {
                T t = T(0.0);
                for(int i3 = 0; i3 < K; i3++)
                {
                    t += T(A[i1 * As1 + i3 * As2]) * T(B[i3 * Bs1 + i2 * Bs2]);
                }
                D[i1 * Ds1 + i2 * Ds2] = beta * C[i1 * Cs1 + i2 * Cs2] + alpha * t;
            }
        }
    }

    template <typename T, typename U = T, typename V = U>
    void matMatMultMixPrec(T   alpha,
                           T   beta,
                           int M,
                           int N,
                           int K,
                           U*  A,
                           int As1,
                           int As2,
                           U*  B,
                           int Bs1,
                           int Bs2,
                           V*  C,
                           int Cs1,
                           int Cs2,
                           V*  D,
                           int Ds1,
                           int Ds2)
    {
        for(int i1 = 0; i1 < M; i1++)
        {
            for(int i2 = 0; i2 < N; i2++)
            {
                T t = T(0.0);
                for(int i3 = 0; i3 < K; i3++)
                {
                    t += T(A[i1 * As1 + i3 * As2]) * T(B[i3 * Bs1 + i2 * Bs2]);
                }
                D[i1 * Ds1 + i2 * Ds2] = V(beta * T(C[i1 * Cs1 + i2 * Cs2]) + alpha * t);
            }
        }
    }

    template <typename T>
    void matIdentity(T* A, int M, int N, size_t lda)
    {
        for(int i = 0; i < M; i++)
        {
            for(int j = 0; j < N; j++)
            {
                A[i + j * lda] = T(i == j);
            }
        }
    }

    template <>
    void matIdentity(hipFloatComplex* A, int M, int N, size_t lda)
    {
        for(int i = 0; i < M; i++)
        {
            for(int j = 0; j < N; j++)
            {
                A[i + j * lda] = hipFloatComplex(i == j, 0);
            }
        }
    }

} // namespace helpers
