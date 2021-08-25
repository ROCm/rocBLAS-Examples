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
SOURCES = $(wildcard *.c) 
OBJECTS = $(patsubst %.c, %.o, $(SOURCES))

CC=gcc
# uncomment to use hip compiler
#CC=$(HIPCXX)
OPT = -ggdb -O0 -march=native -Wall
INC = -isystem$(HIP_PATH)/include -I$(ROCBLAS_PATH)/include -isystem$(ROCM_PATH)/include -I../../common
CCFLAGS = -std=c11 $(INC) $(OPT) 
ifneq ($(CC),$(HIPCXX))
	CCFLAGS += -D__HIP_PLATFORM_AMD__
endif

LDFLAGS=-L$(ROCBLAS_PATH)/lib -lrocblas -Wl,-rpath=$(ROCBLAS_PATH)/lib -L$(ROCM_PATH)/lib -lm -lpthread -lc
ifneq ($(CXX),$(HIPCXX))
	LDFLAGS += -L$(HIP_PATH)/lib -lamdhip64 -Wl,-rpath=$(HIP_PATH)/lib
endif

RM = rm -f

.PHONY: all clean run

all: $(EXE)

%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@

$(EXE): $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@

clean:
	$(RM) $(EXE) $(OBJECTS)

run:
	./$(EXE)

