
# folders to recurse into
DIRS     = $(wildcard */)
DIRSPATH = ./

.PHONY: all clean exe run
 
all: exe

clean: 
	$(foreach dir,$(DIRS),make clean --no-print-directory -C $(DIRSPATH)$(dir);)

exe:
	$(foreach dir,$(DIRS),make -C $(DIRSPATH)$(dir);)

run:
	$(foreach dir,$(DIRS),make run -C $(DIRSPATH)$(dir);)
