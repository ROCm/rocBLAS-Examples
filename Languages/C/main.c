/*
Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.

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

#include <hip/hip_runtime.h>
#include <rocblas.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>


int main(int argc, char** argv)
{
    size_t lda, ldb, lddev;
    size_t rows, cols;

    int n = 10;
    if (argc > 1) n = atoi(argv[1]);

    // { M:  26700, N:   12162,  lda:  26700, ldb:  26700, lddev:  26700 }
    rows = 26700;
    cols = 12162;
    lda = ldb = lddev = 26700;


    typedef double data_type;

    rocblas_handle handle;
    rocblas_status rstatus = rocblas_create_handle(&handle);

    hipStream_t test_stream;
    rocblas_get_stream( handle, &test_stream );

    data_type* ha;
    data_type* hb;
    // allocate pinned memory to allow async memory transfer
    assert( hipMallocHost(&ha, lda * cols * sizeof(data_type)) == hipSuccess);
    assert( hipMallocHost(&hb, lda * cols * sizeof(data_type)) == hipSuccess);

    for(int i1 = 0; i1 < rows; i1++)
        for(int i2 = 0; i2 < cols; i2++)
            ha[i1 + i2 * lda] = 1.0;

    data_type* da = 0;
    data_type* db = 0;
    data_type* dc = 0;
    assert(hipMalloc((void**)&da, lddev * cols * sizeof(data_type)) == hipSuccess);
    assert(hipMalloc((void**)&db, lddev * cols * sizeof(data_type)) == hipSuccess);
    assert(hipMalloc((void**)&dc, lddev * cols * sizeof(data_type)) == hipSuccess);

    rocblas_set_matrix_async(rows, cols, sizeof(data_type), ha, lda, da, lddev, test_stream);
    rocblas_set_matrix_async(rows, cols, sizeof(data_type), ha, lda, db, lddev, test_stream);

    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    //hipStreamSynchronize(test_stream);
    data_type alpha = 1.0;
    data_type beta = 2.0;
    rocblas_dgeam(handle, rocblas_operation_none, rocblas_operation_none, rows, cols, &alpha, da, lddev, &beta, db, lddev, dc, lddev);

    rocblas_get_matrix_async(rows, cols, sizeof(data_type), dc, lddev, hb, ldb, test_stream);

    hipStreamSynchronize(test_stream);

    for(int i1 = 0; i1 < rows; i1++)
        for(int i2 = 0; i2 < cols; i2++)
            assert(hb[i1 + i2 * ldb] == 3.0*ha[i1 + i2 * lda]);

    hipFree(dc);

    // free pinned memory
    hipFreeHost(ha);
    hipFreeHost(hb);

    return 0;
}
