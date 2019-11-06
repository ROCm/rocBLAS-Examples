# rocBLAS-Examples
Examples for using rocBLAS (Radeon Open Compute Basic Linear Algegra Subprograms) which is a GPU exploiting implementation of BLAS. 

# rocBLAS
rocBLAS is AMD's library for [BLAS](http://www.netlib.org/blas/) on [ROCm](https://rocm.github.io/install.html). 
It is implemented in the [HIP](https://github.com/ROCm-Developer-Tools/HIP) 
programming language and optimized for AMD's GPUs.

|Acronym      | Expansion                                                   |
|-------------|-------------------------------------------------------------|
|**BLAS**     | **B**asic **L**inear **A**lgebra **S**ubprograms            |
|**ROCm**     | **R**adeon **O**pen **C**ompute platfor**m**                |
|**HIP**      | **H**eterogeneous-Compute **I**nterface for **P**ortability |

## Documentation
Documentation for each example is contained in the README.md of each example's directory and in the source code itself.
The examples utilize C++ and some shared helper code which is contained in the common/ folder.   The design patterns used in common may be utilized but are intended only to keep the focus of individual examples on the BLAS calling structure.

## Prerequisites
* rocBLAS and it's prerequisites 
If you require rocBLAS it is described at 
[https://github.com/ROCmSoftwarePlatform/rocBLAS](https://github.com/ROCmSoftwarePlatform/rocBLAS)

## Installing
This repository can be cloned into any directory where you want to build the examples. 

## Building
These examples required that you have an installation of rocBLAS on your machine.  You do not required sudo or other access to build these examples which default to gcc but can also use the the hcc compiler from the rocBLAS installation.   The use of hcc can be set by uncommenting lines in the Makefiles.

After cloning this repository you can build all the examples using make in the top level directory, or run make in a sub-level directory to build a specific example:

    cd Level-1/swap 
    make
    ./swap

## Contributing
Additional examples should be added in the Application directory.  The directory name should indicate the application domain and examples must contain a README.md file.   Additional examples may use the common code but should not modify it.  

