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

#include "error_macros.h"
#include <assert.h>
#include <hip/hip_runtime_api.h>
#include <math.h>
#include <rocblas/rocblas.h>
#include <stdio.h>
#include <stdlib.h>

// prototype for external kernel defined in kernel.cpp
__global__ void
    matrix_square_elements(int rows, int cols, const double* a, int lda, double* b, int ldb);

int main(int argc, char** argv)
{
    int lda, ldb, lddev;
    int rows, cols;

    int n = 255;
    int m = 512;
    if(argc > 1)
        n = atoi(argv[1]);
    if(argc > 2)
        m = atoi(argv[2]);

    rows = n;
    cols = m;
    lda = ldb = lddev = n;

    rocblas_handle handle;
    rocblas_status rstatus = rocblas_create_handle(&handle);
    CHECK_ROCBLAS_STATUS(rstatus);

    hipStream_t test_stream;
    rstatus = rocblas_get_stream(handle, &test_stream);
    CHECK_ROCBLAS_STATUS(rstatus);

    double* ha;
    double* hb;

    // allocate pinned memory to allow async memory transfer
    CHECK_HIP_ERROR(hipHostMalloc((void**)&ha, sizeof(double) * lda * cols, hipHostMallocMapped));
    CHECK_HIP_ERROR(hipHostMalloc((void**)&hb, sizeof(double) * ldb * cols, hipHostMallocMapped));

    for(int i1 = 0; i1 < rows; i1++)
        for(int i2 = 0; i2 < cols; i2++)
            ha[i1 + size_t(i2) * lda] = double(i1);

    double* da = 0;
    double* db = 0;
    double* dc = 0;
    CHECK_HIP_ERROR(hipMalloc((void**)&da, sizeof(double) * lddev * cols));
    CHECK_HIP_ERROR(hipMalloc((void**)&db, sizeof(double) * lddev * cols));
    CHECK_HIP_ERROR(hipMalloc((void**)&dc, sizeof(double) * lddev * cols));

    // upload asynchronously from pinned memory
    rstatus = rocblas_set_matrix_async(rows, cols, sizeof(double), ha, lda, da, lddev, test_stream);
    rstatus = rocblas_set_matrix_async(rows, cols, sizeof(double), ha, lda, dc, lddev, test_stream);

    // compute db as square of ha with hip kernel
    const unsigned threads = 32;
    const unsigned rblock  = (rows - 1) / threads + 1;
    const unsigned cblock  = (cols - 1) / threads + 1;
    hipLaunchKernelGGL((matrix_square_elements), /* compute kernel*/
                       dim3(cblock, rblock),
                       dim3(threads, threads),
                       0 /*dynamic shared*/,
                       0 /*stream*/,
                       rows,
                       cols,
                       dc,
                       lda,
                       db,
                       ldb); /* arguments to the compute kernel */

    // this should result in db matrix having each element the squared element value of ha

    // scalar arguments will be from host memory.
    rstatus = rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    CHECK_ROCBLAS_STATUS(rstatus);

    double alpha = 2.0;
    double beta  = 1.0;

    // invoke asynchronous computation
    rstatus = rocblas_dgeam(handle,
                            rocblas_operation_none,
                            rocblas_operation_none,
                            rows,
                            cols,
                            &alpha,
                            da,
                            lddev,
                            &beta,
                            db,
                            lddev,
                            dc,
                            lddev);
    CHECK_ROCBLAS_STATUS(rstatus);

    // fetch results asynchronously to pinned memory
    rstatus = rocblas_get_matrix_async(rows, cols, sizeof(double), dc, lddev, hb, ldb, test_stream);
    CHECK_ROCBLAS_STATUS(rstatus);

    // wait on transfer to be finished
    CHECK_HIP_ERROR(hipStreamSynchronize(test_stream));

    // check against expected results
    bool fail = false;
    for(int i1 = 0; i1 < rows; i1++)
        for(int i2 = 0; i2 < cols; i2++)
        {
            double v = ha[i1 + size_t(i2) * lda];
            if(hb[i1 + size_t(i2) * ldb] != 2.0 * v + v * v)
                fail = true;
        }

    CHECK_HIP_ERROR(hipFree(da));
    CHECK_HIP_ERROR(hipFree(db));
    CHECK_HIP_ERROR(hipFree(dc));

    // free pinned memory
    CHECK_HIP_ERROR(hipHostFree(ha));
    CHECK_HIP_ERROR(hipHostFree(hb));

    rstatus = rocblas_destroy_handle(handle);
    CHECK_ROCBLAS_STATUS(rstatus);

    fprintf(stdout, "%s\n", fail ? "FAIL" : "PASS");

    return 0;
}
