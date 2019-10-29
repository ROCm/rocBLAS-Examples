# rocBLAS-Examples swap
Example showing moving vector data to device and calling the rocblas swap element function. Results are retrieved to host and displayed.

## Documentation
Default values will run the example without any command line arguments.
Running with --help will show:
Usage: ./example
  --n <value>              Size of vector
  --xinc <value>           Increment for x vector
  --yinc <value>           Increment for y vector

## Building
These examples required that you have an installation of rocBLAS on your machine.  You do not required sudo or other access to build these examples which default to gcc but can also use the the hcc compiler from the rocBLAS installation.   The use of hcc can be set by uncommenting lines in the Makefiles.  If rocBLAS is not installed you can set the environment variable ROCBLAS_PATH to point to the location of your rocblas build. 

    cd Level-1/swap 
    make
    ./example
 