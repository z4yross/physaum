CC = nvcc

FILE = main.cu
NAME = physarum

LINKER_FLAGS = glad.o -lGL -lglfw -lm

all:
	rm -f $(NAME)
	$(CC) -o $(NAME) $(FILE) $(LINKER_FLAGS)