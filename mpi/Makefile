
CC = mpiCC
# CFLAGS = 
SRCS = main.cpp
PROG = main


LINKER_FLAGS = glad.o -lGL -lglfw -lm

all:
	rm -f $(PROG)
	$(CC) $(SRCS) -o $(PROG) $(LINKER_FLAGS)