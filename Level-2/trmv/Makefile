# ########################################################################
# Copyright (C) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# ########################################################################

ROCM_PATH?= $(wildcard /opt/rocm)
ifeq (,$(ROCM_PATH))
	ROCM_PATH=
endif

HIP_PATH?= $(wildcard /opt/rocm)
ifeq (,$(HIP_PATH))
	HIP_PATH=
endif
HIPCXX=$(HIP_PATH)/bin/hipcc

ifeq (,$(ROCBLAS_PATH))
	# default to rocblas in standard ROCM tree
	ROCBLAS_PATH= $(ROCM_PATH)
endif

EXE = $(shell basename $(CURDIR))
COMMON_PATH = ../../common
SOURCES = $(wildcard *.cpp) $(wildcard $(COMMON_PATH)/*.cpp)
OBJECTS = $(patsubst %.cpp, %.o, $(SOURCES)) 

CXX=g++
# uncomment to use hip compiler
#CXX=$(HIPCXX)
OPT = -g -Ofast -march=native -Wall
INC = -I$(COMMON_PATH) -isystem$(ROCBLAS_PATH)/include -isystem$(ROCM_PATH)/include 
CXXFLAGS = -std=c++14 $(INC) $(OPT) 
ifneq ($(CXX),$(HIPCXX))
	CXXFLAGS += -D__HIP_PLATFORM_AMD__
endif

LDFLAGS=-L$(ROCBLAS_PATH)/lib -L$(ROCM_PATH)/lib -lrocblas -Wl,-rpath=$(ROCBLAS_PATH)/lib -Wl,-rpath=$(ROCM_PATH)/lib -lm -lpthread -lstdc++
ifneq ($(CXX),$(HIPCXX))
	LDFLAGS += -lamdhip64
endif

RM = rm -f

.PHONY: all clean run

all: $(EXE)

%.o: %.cpp 
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(EXE): $(OBJECTS)
	$(CXX) $(OBJECTS) $(LDFLAGS) -o $@

clean:
	$(RM) $(EXE) $(OBJECTS)

run:
	./$(EXE)

