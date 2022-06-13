# rocBLAS-Examples
Examples for using rocBLAS which is a GPU exploiting implementation of BLAS.

# rocBLAS
rocBLAS is AMD's library for BLAS on [ROCm<sup>TM</sup>](https://rocmdocs.amd.com/en/latest/).
It is implemented in the [HIP](https://github.com/ROCm-Developer-Tools/HIP)
programming language and optimized for AMD's GPUs.

|Acronym      | Expansion                                                   |
|-------------|-------------------------------------------------------------|
|**BLAS**     | **B**asic **L**inear **A**lgebra **S**ubprograms            |
|**HIP**      | **H**eterogeneous-Compute **I**nterface for **P**ortability |

## Documentation
Documentation for each example is contained in the README.md of each example's directory and in the source code itself.
The examples utilize C++ and some shared helper code which is all contained in the common directory.   The design patterns used in common may be utilized but are intended to keep the focus of individual examples on the rocBLAS calling structure.

## Prerequisites
* rocBLAS and it's prerequisites
* ROCm version 3.5 or later (rocBLAS version 2.22 or later)
* As this repo is not tied to specific ROCm releases we recommend building against the latest release of ROCm or the master branch of rocBLAS

If you require rocBLAS it is available at
[https://github.com/ROCmSoftwarePlatform/rocBLAS](https://github.com/ROCmSoftwarePlatform/rocBLAS)

## Installing
This repository can be cloned into any directory where you want to build the examples.

## Building
These examples require that you have an installation of rocBLAS on your machine.  You do not require sudo or other access to build these examples which default to compile with gcc but can also use the hipcc compiler from the rocBLAS installation.  The compiler must support the c++14 standard. The use of hipcc can be set by uncommenting a line in the Makefiles.  The Makefiles support building against a locally built but not installed version of rocBLAS by setting the environment variable ROCBLAS_PATH, e.g.
<tt>export ROCBLAS_PATH=/...yourlocalpath.../rocBLAS/build/release/rocblas-install</tt>

After cloning this repository you can build all the examples using make in the top-level directory, or run make in a sub-level directory to build a specific example:

    cd Level-1/swap
    make
    ./swap

Level-1/swap is the simplest example and is a good starting point to read over the code as it introduces the concepts which may be skipped over in other examples.

Note when compiling with gcc we are defining both the newer <tt>-D__HIP_PLATFORM_AMD__</tt> and the deprecated <tt>-D__HIP_PLATFORM_HCC__</tt> to allow building against various rocm releases.

## Contributing
Additional examples should be added in the Applications directory. The directory name should indicate the application domain and examples must contain a README.md file.   Additional examples may use the common code but should not modify it.

