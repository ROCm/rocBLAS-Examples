# Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.

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
        ROCBLAS_PATH= $(wildcard /opt/rocm/rocblas)
endif

EXE = $(shell basename $(CURDIR))
COMMON_PATH = ../../common
SOURCES = $(wildcard *.cpp) $(wildcard $(COMMON_PATH)/*.cpp)
OBJECTS = $(patsubst %.cpp, %.o, $(SOURCES)) 

CXX=g++
# uncomment to use hip compiler
#CXX=$(HIPCXX)
OPT = -g -Ofast -march=native -Wall
INC = -I$(COMMON_PATH) -isystem$(HIP_PATH)/include -I$(ROCBLAS_PATH)/include -isystem$(ROCM_PATH)/include 
CXXFLAGS = -std=c++14 $(INC) $(OPT) 
ifneq ($(CXX),$(HIPCXX))
	CXXFLAGS += -D__HIP_PLATFORM_HCC__
endif

LDFLAGS=-L$(ROCBLAS_PATH)/lib -lrocblas -Wl,-rpath=$(ROCBLAS_PATH)/lib -L$(ROCM_PATH)/lib -lm -lpthread -lstdc++
ifneq ($(CXX),$(HIPCXX))
	LDFLAGS += -L$(HIP_PATH)/lib -lamdhip64 -Wl,-rpath=$(HIP_PATH)/lib
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

