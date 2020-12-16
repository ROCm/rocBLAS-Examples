# Copyright 2019-2020 Advanced Micro Devices, Inc. All rights reserved.


EXE = $(shell basename $(CURDIR))
SOURCES = ./CMakeLists.txt ./src/CMakeLists.txt ./Makefile-run-cmake
BUILD = build

RM = rm -rf 


.PHONY: all clean run

all: $(BUILD)

$(BUILD): $(SOURCES)
	mkdir -p $(BUILD)
	cp Makefile-run-cmake $(BUILD)
	make -C $(BUILD) -f Makefile-run-cmake

clean:
	$(RM) $(BUILD)

run:
	./$(BUILD)/src/example-cmake

