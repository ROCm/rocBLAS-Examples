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

#include "ArgParser.hpp"
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

using namespace std;

namespace helpers
{

    ArgParserBase::ArgParserBase(const std::vector<std::string>& params)
    {
        addParams(params);
    }

    void ArgParserBase::addParams(const std::vector<std::string>& options)
    {
        if(options.empty())
            return;

        for(auto i : options)
        {
            if(!i.empty())
            {
                decodeOption(i);
            }
        }
    }

    void ArgParserBase::decodeOption(const std::string& option)
    {
        stringstream   ss(option);
        string         item;
        vector<string> parts;
        while(getline(ss, item, '|'))
        {
            if(!item.length())
                return;
            parts.push_back(item);
        }
        if(parts.size() != 3)
            return;

        ArgInfo info;
        info.mParamKeyInt      = (int)(parts[0][0]);
        string key             = parts[1];
        info.mParamDescription = parts[2];

        mOptions[key] = info;
    }

    std::string ArgParserBase::argKey(int argIdx)
    {
        string key(mArgv[argIdx]);
        size_t start = key.find_first_not_of("-");
        return (start == std::string::npos) ? "" : key.substr(start);
    }

    int ArgParserBase::getOptionInt(int argIdx)
    {
        string key(argKey(argIdx));
        if(mOptions.count(key))
        {
            return mOptions[key].mParamKeyInt;
        }
        else
        {
            // see if it matches short form
            if(key.length() == 1)
            {
                for(const auto& i : mOptions)
                {
                    if(i.second.mParamKeyInt == (int)key[0])
                        return (int)key[0];
                }
            }
            return -1;
        }
    }

    void ArgParserBase::printHelp()
    {
        cout << "Usage: " << mArgv[0] << std::endl;
        if(!mOptions.size())
            cout << "Options:" << std::endl;

        for(const auto& i : mOptions)
        {
            string arg("  --");
            arg += i.first + " <value>";
            while(arg.length() < 27)
                arg += " ";
            cout << arg << i.second.mParamDescription << endl;
        }
    }

    void ArgParserBase::usage(int argIdx)
    {
        if(argIdx != 0 && argIdx < mArgc)
        {
            string key = argKey(argIdx);
            if(mOptions.count(key))
            {
                std::cout << "Usage: Bad or missing argument for " << key << ": "
                          << mOptions[key].mParamDescription << std::endl;
                if(argIdx + 1 < mArgc)
                    std::cout << "  value was: " << mArgv[argIdx + 1] << std::endl;
            }
            else
            {
                printHelp();
            }
        }
        else
        {
            printHelp();
        }
    }

    bool ArgParserBase::validArgs(int argc, char** argv)
    {
        mArgc = argc;
        mArgv = argv;

        int argIdx = 1;
        while(argIdx < argc)
        {
            int status = parseStandardOption(argIdx, argv);
            if(status == eNoMatch)
            {
                if(!parse(argIdx, argv))
                {
                    usage(argIdx);
                    return false;
                }
            }
            else if(status == eInvalid)
            {
                usage(argIdx);
                return false;
            }
        }
        cout << "Parsed options for: " << mArgv[0] << std::endl;
        return true;
    }

    //
    // Class for blas examples and common parameters
    //
    ArgParser::ArgParser(std::string standardArgs, const std::vector<std::string>& params)
        : ArgParserBase::ArgParserBase(params)
    {
        static const char* const cn = "n|n|Size of vector";
        static const char* const cx = "x|incx|Increment for x vector";
        static const char* const cy = "y|incy|Increment for y vector";
        static const char* const cM = "M|M|Matrix/vector dimension";
        static const char* const cN = "N|N|Matrix/vector dimension";
        static const char* const cK = "K|K|Matrix/vector dimension";
        static const char* const ca = "a|alpha|Alpha scalar";
        static const char* const cb = "b|beta|Beta scalar";
        static const char* const cc = "c|count|Batch count";

        std::vector<std::string> stdArgs;
        for(size_t i = 0; i < standardArgs.length(); i++)
        {
            switch(standardArgs[i])
            {
            case 'n':
                stdArgs.push_back(string(cn));
                break;
            case 'x':
                stdArgs.push_back(string(cx));
                break;
            case 'y':
                stdArgs.push_back(string(cy));
                break;
            case 'a':
                stdArgs.push_back(string(ca));
                break;
            case 'b':
                stdArgs.push_back(string(cb));
                break;
            case 'c':
                stdArgs.push_back(string(cc));
                break;
            case 'M':
                stdArgs.push_back(string(cM));
                break;
            case 'N':
                stdArgs.push_back(string(cN));
                break;
            case 'K':
                stdArgs.push_back(string(cK));
                break;
            default:
                break;
            }
        }

        if(stdArgs.size())
            addParams(stdArgs);
    }

    int ArgParser::parseStandardOption(int& argIdx, char** argv)
    {
        int key = getOptionInt(argIdx++);

        // all standard options have arguments
        if(key < 0 || argIdx >= mArgc)
        {
            --argIdx;
            return eInvalid;
        }

        bool found = true;
        try
        {
            switch(key)
            {
            case 'x':
                incx = atoi(argv[argIdx++]);
                break;
            case 'y':
                incy = atoi(argv[argIdx++]);
                break;
            case 'n':
                n = atoi(argv[argIdx++]);
                break;
            case 'a':
                alpha = (float)atof(argv[argIdx++]);
                break;
            case 'b':
                beta = (float)atof(argv[argIdx++]);
                break;
            case 'c':
                batchCount = atoi(argv[argIdx++]);
                break;
            case 'M':
                M = atoi(argv[argIdx++]);
                break;
            case 'N':
                N = atoi(argv[argIdx++]);
                break;
            case 'K':
                K = atoi(argv[argIdx++]);
                break;
            default:
                found = false;
                break;
            }
        }
        catch(...)
        {
            return eInvalid;
        }

        return found ? eOk : eNoMatch;
    }
} // namespace helpers
