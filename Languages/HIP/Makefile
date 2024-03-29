# Copyright (C) 2021-2023 Advanced Micro Devices, Inc. All rights reserved.

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
SOURCES = $(wildcard *.cpp) 
OBJECTS = $(patsubst %.cpp, %.o, $(SOURCES))

CXX=$(HIPCXX) # needed for hip kernel compilation
OPT = -ggdb -O0 -march=native -Wall
INC = -isystem$(ROCBLAS_PATH)/include -isystem$(ROCM_PATH)/include -I../../common
CXXFLAGS = -std=c++14 $(INC) $(OPT)
ifneq ($(CXX),$(HIPCXX))
	CCFLAGS += -D__HIP_PLATFORM_AMD__ 
endif

LDFLAGS=-L$(ROCBLAS_PATH)/lib -L$(ROCM_PATH)/lib -lrocblas -Wl,-rpath=$(ROCBLAS_PATH)/lib -Wl,-rpath=$(ROCM_PATH)/lib -lm -lpthread -lstdc++
ifneq ($(CXX),$(HIPCXX))
	LDFLAGS += -lamdhip64
endif

RM = rm -f

.PHONY: all clean run

all: $(EXE)

%.o: %.cpp # all are using hip
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(EXE): $(OBJECTS)
	$(CXX) $(OBJECTS) $(LDFLAGS) -o $@

clean:
	$(RM) $(EXE) $(OBJECTS)

run:
	./${EXE}
	# to log hip function calls
	# AMD_LOG_LEVEL=3 ./$(EXE)

