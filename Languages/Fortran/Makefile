# ########################################################################
# Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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

HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
        HIP_PATH=
endif
HIPCXX=$(HIP_PATH)/bin/hipcc

ifeq (,$(ROCBLAS_PATH))
        ROCBLAS_PATH= $(wildcard /opt/rocm/include)
endif

EXE = $(shell basename $(CURDIR))
SOURCES = $(wildcard *.f90) 
OBJECTS = $(patsubst %.f90, %.o, $(SOURCES))

CC=gfortran
OPT = -ggdb -O0 -march=native -Wall -Wno-c-binding-type
INC = -isystem$(ROCBLAS_PATH)/include -isystem$(ROCM_PATH)/include -I../../common
FFLAGS = $(INC) $(OPT) 
ifneq ($(CC),$(HIPCXX))
	FFLAGS += -D__HIP_PLATFORM_AMD__ -D__HIP_PLATFORM_HCC__
endif

LDFLAGS=-L$(ROCBLAS_PATH)/lib -L$(ROCM_PATH)/lib -lrocblas -Wl,-rpath=$(ROCBLAS_PATH)/lib -Wl,-rpath=$(ROCM_PATH)/lib -lm -lpthread -lstdc++
ifneq ($(CXX),$(HIPCXX))
	LDFLAGS += -lamdhip64
endif

RM = rm -f

.PHONY: all clean run

all: $(EXE)

%.o: %.f90
	$(CC) $(FFLAGS) -c $< -o $@

rocblas.mod: $(ROCBLAS_PATH)/rocblas/rocblas_module.f90
	$(CC) $(FFLAGS) -c $<

$(EXE): rocblas.mod $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@

clean:
	$(RM) $(EXE) $(OBJECTS) *.mod rocblas_module.o

run:
	./$(EXE)

