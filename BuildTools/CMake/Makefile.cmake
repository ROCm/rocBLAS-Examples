# Copyright 2019-2020 Advanced Micro Devices, Inc. All rights reserved.

ROCM_PATH?= $(wildcard /opt/rocm)
ifeq (,$(ROCM_PATH))
        ROCM_PATH=
endif

CXX=g++
#CXX=hipcc

.PHONY: runcmake

runcmake:
	CXX=$(CXX) cmake -DCMAKE_PREFIX_PATH=$(ROCM_PATH) ..
	make

