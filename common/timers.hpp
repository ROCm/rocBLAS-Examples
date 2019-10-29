/*
Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.

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
#include <chrono>
#include <hip/hip_runtime.h>

namespace helpers
{

    class GPUTimer
    {
    public:
        GPUTimer()
        {
            hipEventCreate(&mStart);
            hipEventCreate(&mStop);
        }
        virtual ~GPUTimer() {}

        void start()
        {
            hipEventRecord(mStart);
        }
        float stop(const char* msg = nullptr)
        {
            hipEventRecord(mStop);
            hipEventSynchronize(mStop);
            float time_elapsed = 0.0f;
            hipEventElapsedTime(&time_elapsed, mStart, mStop);
            const char* prefix = msg ? msg : "HipEventTime elpased: ";
            std::cout << prefix << time_elapsed << "ms" << std::endl;
            return time_elapsed;
        }

    protected:
        hipEvent_t mStart;
        hipEvent_t mStop;
    };

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
            mStop = std::chrono::high_resolution_clock::now();
            double time_elapsed
                = std::chrono::duration_cast<std::chrono::nanoseconds>(mStop - mStart).count()
                  * 1e-6;
            const char* prefix = msg ? msg : "Time elpased: ";
            std::cout << prefix << time_elapsed << "ms" << std::endl;
            return time_elapsed;
        }

    protected:
        std::chrono::high_resolution_clock::time_point mStart;
        std::chrono::high_resolution_clock::time_point mStop;
    };

} // namespace helpers