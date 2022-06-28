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
#include "error_macros.h"
#include <chrono>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <iostream>

namespace helpers
{

    /*! \brief  Hip event based GPU timer
    ********************************************************************/
    class GPUTimer
    {
    public:
        GPUTimer()
        {
            CHECK_HIP_ERROR(hipEventCreate(&mStart));
            CHECK_HIP_ERROR(hipEventCreate(&mStop));
        }
        virtual ~GPUTimer()
        {
            CHECK_HIP_ERROR(hipEventDestroy(mStart));
            CHECK_HIP_ERROR(hipEventDestroy(mStop));
        }

        void start()
        {
            CHECK_HIP_ERROR(hipEventRecord(mStart));
        }
        float stop(const char* msg = nullptr)
        {
            CHECK_HIP_ERROR(hipEventRecord(mStop));
            CHECK_HIP_ERROR(hipEventSynchronize(mStop));
            float timeElaspsedMillisec = 0.0f;
            CHECK_HIP_ERROR(hipEventElapsedTime(&timeElaspsedMillisec, mStart, mStop));
            const char* prefix = msg ? msg : "hipEventElapsedTime: ";
            std::cout << prefix << timeElaspsedMillisec << "ms" << std::endl;
            return timeElaspsedMillisec;
        }

    protected:
        hipEvent_t mStart;
        hipEvent_t mStop;
    };

    /*! \brief  std::chrono based CPU timer
    ********************************************************************/
    class CPUTimer
    {
    public:
        CPUTimer() {}
        virtual ~CPUTimer() {}

        void start()
        {
            mStart = std::chrono::high_resolution_clock::now();
        }
        double stop(const char* msg = nullptr)
        {
            mStop                               = std::chrono::high_resolution_clock::now();
            constexpr double cNanosecToMillisec = 1e-6;
            double           timeElaspsedMillisec
                = std::chrono::duration_cast<std::chrono::nanoseconds>(mStop - mStart).count()
                  * cNanosecToMillisec;
            const char* prefix = msg ? msg : "Time elpased: ";
            std::cout << prefix << timeElaspsedMillisec << "ms" << std::endl;
            return timeElaspsedMillisec;
        }

    protected:
        std::chrono::high_resolution_clock::time_point mStart;
        std::chrono::high_resolution_clock::time_point mStop;
    };

} // namespace helpers
