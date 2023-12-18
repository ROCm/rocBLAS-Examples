# Change Log for rocBLAS-Examples

Full documentation for rocBLAS is available at [rocblas.readthedocs.io](https://rocblas.readthedocs.io/en/latest/).

## For rocBLAS 4.1.0 in ROCm 6.1.0

### Changed
- gemm_ex examples t0 use transpose arguments NT

## For rocBLAS 4.0.0 in ROCm 6.0.0

### Changed
- Fixed tolerance for half precisions to include problem size accumulation factor

## For rocBLAS 3.1.0 in ROCm 5.7.0

### Changed
- Added example calling complex version of trmv (ctrmv) which illustrates how different data types for handling complex numbers can be used and also absolute and relative error checking.
- Removed the deprecated paths to hip/bin and rocblas fortran module being removed in future releases

## For rocBLAS 3.0.0 in ROCm 5.6.0

### Changed
- Removed deprecated defines, code and referring to deprecated folders

## For rocBLAS 2.47.0 in ROCm 5.5.0

### Fixed
- On Windows the Visual Studio toolchain include of rocblas.h shows deprecation warnings. No longer requires define ROCBLAS_NO_DEPRECATED_WARNINGS.

## For rocBLAS 2.46.0 in ROCm 5.4.0

### Changed
- On Windows the Visual Studio toolchain include of rocblas.h requires adding a define ROCBLAS_NO_DEPRECATED_WARNINGS for ROCM 5.4

## For rocBLAS 2.45.0 in ROCm 5.3.0

### Changed
- No changes were made for this release.

## For rocBLAS 2.44.0 in ROCm 5.2.0

### Changed
- ROCm<sup>TM</sup> installation paths for include files and libraries have changed locations.  This release of examples takes these from the new locations to avoid the deprecation messages introduced into the old header locations.  Relevant changes can be seen in the Makefiles.

