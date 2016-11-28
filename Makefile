TARGETS = rectify pool convolve test grid_4_4 grid_512_512
LIBS = -lm
CC = nvcc

.PHONY: default all clean

default: $(TARGETS)
all: default

pool:
        nvcc pooling.cu lodepng.cu -o pool -lm

rectify:
        nvcc rectify.cu lodepng.cu -o rectify -lm
        
convolve:
        nvcc convolution.cu lodepng.cu -o convolve -lm

grid_4_4:
        nvcc grid_4_4.cu -o grid_4_4

grid_512_512:
        nvcc grid_512_512.cu -o grid_512_512

clean:
        -rm -f *.o
        -rm -f $(TARGETS)