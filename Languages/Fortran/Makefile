# Copyright 2019-2020 Advanced Micro Devices, Inc. All rights reserved.

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
SOURCES = $(wildcard *.f90) 
OBJECTS = $(patsubst %.f90, %.o, $(SOURCES))

CC=gfortran
OPT = -ggdb -O0 -march=native -Wall -Wno-c-binding-type
INC = -isystem$(HIP_PATH)/include -I$(ROCBLAS_PATH)/include -isystem$(ROCM_PATH)/include -I../../common
FFLAGS = $(INC) $(OPT) 
ifneq ($(CC),$(HIPCXX))
	FFLAGS += -D__HIP_PLATFORM_HCC__
endif

LDFLAGS=-L$(ROCBLAS_PATH)/lib -lrocblas -Wl,-rpath=$(ROCBLAS_PATH)/lib -L$(ROCM_PATH)/lib -lm -lpthread
ifneq ($(CXX),$(HIPCXX))
	LDFLAGS += -L$(HIP_PATH)/lib -lamdhip64 -Wl,-rpath=$(HIP_PATH)/lib
endif

RM = rm -f

.PHONY: all clean run

all: $(EXE)

%.o: %.f90
	$(CC) $(FFLAGS) -c $< -o $@

rocblas.mod: $(ROCBLAS_PATH)/include/rocblas_module.f90
	$(CC) $(FFLAGS) -c $<

$(EXE): rocblas.mod $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@

clean:
	$(RM) $(EXE) $(OBJECTS) *.mod rocblas_module.o

run:
	./$(EXE)

