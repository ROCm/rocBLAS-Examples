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

#pragma once
#include "ArgParser.hpp"
#include "error_macros.h"
#include "memoryHelpers.hpp"
#include "timers.hpp"
#include <complex>
#include <cstdio>
#if !defined(_WIN32) || defined(__HIPCC__)
#include <hip/hip_complex.h>
#endif
#include <iostream>
#include <random>
#include <rocblas/rocblas.h>
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
        (void)distrib(gen); // prime generator to remove warning

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

#if !defined(_WIN32) || defined(__HIPCC__)
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
        (void)distrib(gen); // prime generator to remove warning

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
        (void)distrib(gen); // prime generator to remove warning

        for(size_t i = 0; i < arr.size(); i += inc)
        {
            int rval = distrib(gen);
            int ival = distrib(gen);
            arr[i]   = T((float)rval, (float)ival);
        }
    }

    template <typename T,
              std::enable_if_t<!std::is_same<T, std::complex<float>>{}
                                   && !std::is_same<T, hipFloatComplex>{},
                               int> = 0>
    void fillVectorUniformRealDist(std::vector<T>& arr,
                                   float           lower_range = -3.0,
                                   float           upper_range = 3.0)
    {
        srand(int(time(NULL)));
        std::random_device                    rd{};
        std::mt19937                          gen{rd()};
        std::uniform_real_distribution<float> distrib{lower_range, upper_range};
        (void)distrib(gen); // prime generator to remove warning

        for(size_t i = 0; i < arr.size(); i++)
        {
            float val = distrib(gen);
            arr[i]    = T((float)val);
        }
    }

    template <typename T,
              std::enable_if_t<std::is_same<T, std::complex<float>>{}
                                   || std::is_same<T, hipFloatComplex>{},
                               int> = 0>
    void fillVectorUniformRealDist(std::vector<T>& arr,
                                   float           lower_range = -3.0,
                                   float           upper_range = 3.0)
    {
        srand(int(time(NULL)));
        std::random_device                    rd{};
        std::mt19937                          gen{rd()};
        std::uniform_real_distribution<float> distrib{lower_range, upper_range};
        (void)distrib(gen); // prime generator to remove warning

        for(size_t i = 0; i < arr.size(); i++)
        {
            float rval = distrib(gen);
            float ival = distrib(gen);
            arr[i]     = T((float)rval, (float)ival);
        }
    }

#else
    template <typename T>
    void fillVectorUniformIntRand(std::vector<T>& arr, rocblas_int inc = 1, int range = 3)
    {
        srand(int(time(NULL)));
        std::random_device                 rd{};
        std::mt19937                       gen{rd()};
        std::uniform_int_distribution<int> distrib{-range, range};
        (void)distrib(gen); // prime generator to remove warning

        for(size_t i = 0; i < arr.size(); i += inc)
        {
            int rval = distrib(gen);
            arr[i]   = T((float)rval);
        }
    }

    template <typename T>
    void fillVectorUniformRealDist(std::vector<T>& arr,
                                   float           lower_range = -3.0,
                                   float           upper_range = 3.0)
    {
        srand(int(time(NULL)));
        std::random_device                    rd{};
        std::mt19937                          gen{rd()};
        std::uniform_real_distribution<float> distrib{lower_range, upper_range};
        (void)distrib(gen); // prime generator to remove warning

        for(size_t i = 0; i < arr.size(); i++)
        {
            float val = distrib(gen);
            arr[i]    = T((float)val);
        }
    }

#endif

    template <typename T>
    void makeMatrixUpperOrlower(rocblas_fill uplo, std::vector<T>& A, rocblas_int N, size_t lda)
    {
        //zero out the upper part
        if(uplo == rocblas_fill_lower)
        {
            for(int col = 1; col < N; col++)
            {
                for(int row = 0; row < col; row++)
                {
                    A[col * lda + row] = T(0.0);
                }
            }
        }
        //zero out the lower part
        else
        {
            for(int col = 0; col < N; col++)
            {
                for(int row = col + 1; row < N; row++)
                {
                    A[col * lda + row] = T(0.0);
                }
            }
        }
    }

    template <typename T>
    void make_unit_diagonal(rocblas_fill uplo, std::vector<T>& A, rocblas_int N, size_t lda)
    {
        if(uplo == rocblas_fill_lower)
        {
            for(int i = 0; i < N; i++)
            {
                T diag = A[i + i * lda];
                for(int j = 0; j <= i; j++)
                    A[i + j * lda] = A[i + j * lda] / diag;
            }
        }
        else // rocblas_fill_upper
        {
            for(int j = 0; j < N; j++)
            {
                T diag = A[j + j * lda];
                for(int i = 0; i <= j; i++)
                    A[i + j * lda] = A[i + j * lda] / diag;
            }
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
    double maxRelativeErrorComplexVector(std::vector<T>& A,
                                         std::vector<T>& reference,
                                         rocblas_int     N,
                                         size_t          incx)
    {
        double real_maxRelativeError = double(std::numeric_limits<float>::min());
        double imag_maxRelativeError = double(std::numeric_limits<float>::min());

        for(int i = 0; i < N; i++)
        {
            if(std::real(reference[i * incx]) != std::real(A[i * incx]))
            {
                double gold          = double(std::real(reference[i * incx]));
                double relativeError = gold != 0 ? (gold - double(std::real(A[i * incx]))) / (gold)
                                                 : double(std::real(A[i * incx]));
                relativeError        = relativeError > 0 ? relativeError : -relativeError;
                real_maxRelativeError
                    = relativeError < real_maxRelativeError ? real_maxRelativeError : relativeError;
            }
            if(std::imag(reference[i * incx]) != std::imag(A[i * incx]))
            {
                double gold          = double(std::imag(reference[i * incx]));
                double relativeError = gold != 0 ? (gold - double(std::imag(A[i * incx]))) / (gold)
                                                 : double(std::imag(A[i * incx]));
                relativeError        = relativeError > 0 ? relativeError : -relativeError;
                imag_maxRelativeError
                    = relativeError < imag_maxRelativeError ? imag_maxRelativeError : relativeError;
            }
        }
        return real_maxRelativeError >= imag_maxRelativeError ? real_maxRelativeError
                                                              : imag_maxRelativeError;
    }

    template <typename T>
    double maxAbsoulteErrorComplexVector(std::vector<T>& A,
                                         std::vector<T>& reference,
                                         rocblas_int     N,
                                         size_t          incx)
    {
        double real_maxAbsoluteError = double(std::numeric_limits<float>::min());
        double imag_maxAbsoluteError = double(std::numeric_limits<float>::min());

        for(int i = 0; i < N; i++)
        {
            if(std::abs(std::real(reference[i * incx]) - std::real(A[i * incx])) > 1.0)
            {
                double AbsoluteError
                    = std::abs(std::real(reference[i * incx]) - std::real(A[i * incx]));
                real_maxAbsoluteError
                    = AbsoluteError < real_maxAbsoluteError ? real_maxAbsoluteError : AbsoluteError;
            }
            if(std::abs(std::imag(reference[i * incx]) - std::imag(A[i * incx])) > 1.0)
            {
                double AbsoluteError
                    = std::abs(std::imag(reference[i * incx]) - std::imag(A[i * incx]));
                imag_maxAbsoluteError
                    = AbsoluteError < imag_maxAbsoluteError ? imag_maxAbsoluteError : AbsoluteError;
            }
        }
        return real_maxAbsoluteError >= imag_maxAbsoluteError ? real_maxAbsoluteError
                                                              : imag_maxAbsoluteError;
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

#if !defined(_WIN32) || defined(__HIPCC__)
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
#endif

} // namespace helpers
