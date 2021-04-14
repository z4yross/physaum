FILE = physarum.c

CC = gcc

LINKER_FLAGS = -lGL -lglfw -lm

NAME = physarum

all: $(FILE) 
	$(CC) $(FILE) $(LINKER_FLAGS) -o $(NAME)