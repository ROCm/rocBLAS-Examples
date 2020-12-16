
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
