ROC_PATH?= $(wildcard /opt/rocm)
ifeq (,$(ROC_PATH))
        ROC_PATH=
endif

HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
        HIP_PATH=
endif
HIPCC=$(HIP_PATH)/bin/hipcc

#HIP_PLATFORM=$(shell $(HIP_PATH)/bin/hipconfig --platform)
ROCBLAS_PATH?= $(wildcard /opt/rocm/rocblas)
ROCBLAS_PATH=/home/torrezuk/blas/build/release/rocblas-install

EXE = example
COMMON_PATH = ../../common
SOURCES = $(wildcard *.cpp) $(wildcard $(COMMON_PATH)/*.cpp)
OBJECTS = $(patsubst %.cpp, %.o, $(SOURCES)) 

CC=g++
# uncomment to use hip compiler
#CC=$(HIPCC)
OPT = -g -Ofast -march=native 
INC = -I$(COMMON_PATH) -isystem$(HIP_PATH)/include -I$(ROCBLAS_PATH)/include -isystem$(ROC_PATH)/include 
CXXFLAGS = -std=c++11 $(INC) $(OPT)
ifneq ($(CC),$(HIPCC))
	CXXFLAGS += -D__HIP_PLATFORM_HCC__ 
endif

LDFLAGS=-L$(ROCBLAS_PATH)/lib -lrocblas -Wl,-rpath=$(ROCBLAS_PATH)/lib -L$(ROC_PATH)/lib -lm -lpthread -lstdc++
ifneq ($(CC),$(HIPCC))
	LDFLAGS += -L$(HIP_PATH)/lib -lhiprtc -lhip_hcc 
endif

RM = rm -f

.PHONY: all clean 

all: $(EXE)

%.o: %.cpp 
	$(CC) $(CXXFLAGS) -c $< -o $@

$(EXE): $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@

clean:
	$(RM) $(EXE) $(OBJECTS)

