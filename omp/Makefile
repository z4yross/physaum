FILE = physarum.cpp

CC = g++
CFLAGS = -g -Wall -fopenmp -lm -D_DEFAULT_SOURCE

LINKER_FLAGS = -lGL -lglfw -lm

NAME = physarum

all: $(FILE) 
	$(CC) $(FILE) $(LINKER_FLAGS) $(CFLAGS) -o $(NAME)