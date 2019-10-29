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

#include "errorHelpers.hpp"

namespace helpers
{
    const char* const rocblasGetStatusString(rocblas_status status)
    {
        switch(status)
        {
        case rocblas_status_success:
            return "rocblas_status_success";
        case rocblas_status_invalid_handle:
            return "rocblas_status_invalid_handle";
        case rocblas_status_not_implemented:
            return "rocblas_status_not_implemented";
        case rocblas_status_invalid_pointer:
            return "rocblas_status_invalid_pointer";
        case rocblas_status_invalid_size:
            return "rocblas_status_invalid_size";
        case rocblas_status_memory_error:
            return "rocblas_status_memory_error";
        case rocblas_status_internal_error:
            return "rocblas_status_internal_error";
        default:
            return "<undefined rocblas_status value>";
        }
    }
} // namespace helpers
