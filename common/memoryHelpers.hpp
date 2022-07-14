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
#include <cinttypes>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <locale.h>
#include <random>
#include <stdlib.h>
#include <vector>

namespace helpers
{

    /*! \brief  base-class to allocate/deallocate device memory */
    template <typename T, size_t PAD, typename U>
    class DeviceVectorMemory
    {
    protected:
        size_t mSize, mBytes;

        U mGuard[PAD];
        DeviceVectorMemory(size_t s)
            : mSize(s)
            , mBytes((s + PAD * 2) * sizeof(T))
        {
            // Initialize mGuard
            if(PAD > 0)
            {
                memset(&mGuard[0], 0xfe, PAD * sizeof(U));
            }
        }

        T* setup()
        {
            T* d;
            if((hipMalloc)(&d, mBytes) != hipSuccess)
            {
                fprintf(stderr, "Error allocating %zu mBytes (%zu GB)\n", mBytes, mBytes >> 30);
                d = nullptr;
            }
            else
            {
                if(PAD > 0)
                {
                    // Copy mGuard to device memory before start of allocated memory
                    CHECK_HIP_ERROR(hipMemcpy(d, mGuard, sizeof(mGuard), hipMemcpyHostToDevice));

                    // Point to allocated block
                    d += PAD;

                    // Copy mGuard to device memory after end of allocated memory
                    CHECK_HIP_ERROR(
                        hipMemcpy(d + mSize, mGuard, sizeof(mGuard), hipMemcpyHostToDevice));
                }
            }
            return d;
        }

        void teardown(T* d)
        {
            if(d != nullptr)
            {
                if(PAD > 0)
                {
                    U host[PAD];

                    // Copy device memory after allocated memory to host
                    CHECK_HIP_ERROR(
                        hipMemcpy(host, d + mSize, sizeof(mGuard), hipMemcpyDeviceToHost));

                    // Make sure no corruption has occurred
                    assert(!memcmp(host, mGuard, sizeof(mGuard)));

                    // Point to mGuard before allocated memory
                    d -= PAD;

                    // Copy device memory after allocated memory to host
                    CHECK_HIP_ERROR(hipMemcpy(host, d, sizeof(mGuard), hipMemcpyDeviceToHost));

                    // Make sure no corruption has occurred
                    assert(!memcmp(host, mGuard, sizeof(mGuard)));
                }
                // Free device memory
                CHECK_HIP_ERROR((hipFree)(d));
            }
        }
    };

    /*! \brief  pseudo-vector subclass which uses device memory */
    template <typename T, size_t PAD = 4096, typename U = T>
    class DeviceVector : private DeviceVectorMemory<T, PAD, U>
    {
    public:
        explicit DeviceVector(size_t s)
            : DeviceVectorMemory<T, PAD, U>(s)
        {
            mData = this->setup();
        }

        ~DeviceVector()
        {
            this->teardown(mData);
        }

        // Decay into pointer wherever pointer is expected
        operator T*()
        {
            return mData;
        }

        operator const T*() const
        {
            return mData;
        }

        T* data() const
        {
            return mData;
        }

        // Tell whether malloc failed
        explicit operator bool() const
        {
            return mData != nullptr;
        }

        // Disallow copying or assigning
        DeviceVector(const DeviceVector&) = delete;
        DeviceVector& operator=(const DeviceVector&) = delete;

    private:
        T* mData;
    };

    /*! \brief  pseudo-vector subclass which uses a BatchCount of device memory
                pointers and an array of pointers in host memory*/
    template <typename T, size_t PAD = 4096, typename U = T>
    class DeviceBatchVector : private DeviceVectorMemory<T, PAD, U>
    {
    public:
        explicit DeviceBatchVector(size_t b, size_t s)
            : mBatchCount(b)
            , DeviceVectorMemory<T, PAD, U>(s)
        {
            mData = (T**)malloc(mBatchCount * sizeof(T*));
            for(int b = 0; b < mBatchCount; ++b)
                mData[b] = this->setup();
        }

        ~DeviceBatchVector()
        {
            if(mData != nullptr)
            {
                for(int b = 0; b < mBatchCount; ++b)
                    this->teardown(mData[b]);
                free(mData);
            }
        }

        T* operator[](int n)
        {
            return mData[n];
        }

        operator T**()
        {
            return mData;
        }

        T** data() const
        {
            return mData;
        }

        // Disallow copying or assigning
        DeviceBatchVector(const DeviceBatchVector&) = delete;
        DeviceBatchVector& operator=(const DeviceBatchVector&) = delete;

    private:
        T**    mData;
        size_t mBatchCount;
    };

} // namespace helpers
