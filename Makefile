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


# folders to recurse into
LIBS     = Level-1 Level-2 Level-3 Extensions Languages Patterns BuildTools
LIBSPATH = ./

.PHONY: all clean exe run
 
all: exe

clean: 
	$(foreach dir,$(LIBS),$(foreach eg, $(wildcard $(dir)/*/), make clean --no-print-directory -C $(LIBSPATH)/$(eg);) )

exe:
	$(foreach dir,$(LIBS),$(foreach eg, $(wildcard $(dir)/*/), make --no-print-directory -C $(LIBSPATH)/$(eg);) )

run:
	$(foreach dir,$(LIBS),$(foreach eg, $(wildcard $(dir)/*/), make run --no-print-directory -C $(LIBSPATH)/$(eg);) )
