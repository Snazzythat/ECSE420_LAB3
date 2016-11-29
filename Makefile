TARGETS = rectify pool convolve grid_4_4 grid_512_512
LIBS = -lm
CC = nvcc

.PHONY: default all clean

default: $(TARGETS)
all: default

pool:
	$(CC) pool.cu lodepng.cu -o pool -lm

rectify:
	$(CC) rectify.cu lodepng.cu -o rectify -lm
        
convolve:
	$(CC) convolve.cu lodepng.cu -o convolve -lm

grid_4_4:
	nvcc grid_4_4.cu -o grid_4_4

#grid_512_512:
	#nvcc grid_512_512.cu -o grid_512_512

clean:
	-rm -f *.o
	-rm -f $(TARGETS)
