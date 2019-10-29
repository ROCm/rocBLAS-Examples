
# folders to recurse into
LIBS     = Level-1 Level-2 Level-3
LIBSPATH = ./

.PHONY: all clean exe run
 
all: exe

clean: 
	$(foreach dir,$(LIBS),make clean --no-print-directory -C $(LIBSPATH)$(dir);)

exe:
	$(foreach dir,$(LIBS),make -C $(LIBSPATH)$(dir);)

run:
	$(foreach dir,$(LIBS),make run -C $(LIBSPATH)$(dir);)
