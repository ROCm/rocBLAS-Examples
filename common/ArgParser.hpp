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

#include <hip/hip_runtime.h>
#include <map>
#include <math.h>
#include <rocblas/rocblas.h>
#include <set>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

namespace helpers
{
    struct ArgInfo
    {
        std::string mParamDescription;
        int         mParamKeyInt;
    };

    /*! \brief  Base class for simple command line parsing
    ********************************************************************/
    class ArgParserBase
    {
    public:
        enum
        {
            eOk,
            eInvalid,
            eNoMatch,
        };

        ArgParserBase(const std::vector<std::string>& options = {});

        void printHelp();

        void usage(int argIdx);

        void decodeOption(const std::string& option);

        void addParams(const std::vector<std::string>& options);

        std::string argKey(int argIdx);

        int getOptionInt(int argIdx);

        virtual int parse(int& argIdx, char** argv)
        {
            return eNoMatch;
        }

        virtual int parseStandardOption(int& argIdx, char** argv)
        {
            return eOk;
        }

        /*! * @brief validArgs parses the command line options
            * @param argc number of elements in cmd line input
            * @param argv array of char* storing the CmdLine Options
            * @return true if all specified arguments are valid
        ********************************************************************/
        bool validArgs(int argc, char** argv);

    protected:
        // arg processing information
        std::map<std::string, ArgInfo> mOptions;
        std::set<std::string>          mStandardParam;

        int    mArgc;
        char** mArgv;
    };

    /*! \brief class for rocblas examples command line parsing of common parameters
    *******************************************************************************/
    class ArgParser : public ArgParserBase
    {
    public:
        ArgParser(std::string standardArgs, const std::vector<std::string>& options = {});

        int parseStandardOption(int& argIdx, char** argv) override;

    public:
        // common arguments for rocblas functions

        rocblas_int M = 5;
        rocblas_int N = 5;
        rocblas_int K = 5;

        rocblas_int n    = 5;
        rocblas_int incx = 1;
        rocblas_int incy = 1;

        rocblas_int batchCount = 3;

        float alpha = 1.0f;
        float beta  = 1.0f;
    };

} // namespace helpers
