
#Flags
CC = gcc
FLAGS = -g -Wall -Wno-unused-function -O3 -march=native -fopenmp -fno-align-loops
CFLAGS += $(FLAGS)
LDFLAGS += $(FLAGS) -L/usr/lib -lm -lpapi -ldnnl
	# -lpthread

C_SRC=mem_utils.c timing.c oneDNN_conv.c main.c
INC=$(wildcard *.h)
OBJ=$(C_SRC:%.c=%.o)

.phony: all clean
all: convDNN

convDNN: $(OBJ) 
	$(CC) $^ $(LDFLAGS) -o $@

%.o: %.c $(INC)
	$(CC) -c $< $(CFLAGS)  -o $@

clean:
	rm -r $(OBJ) convDNN




