#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wm.h"
#include "gputimer.h"

#define GRID_SIZE 512
#define BLCK_SIZ 512
#define P 0.5
#define N 0.0002
#define G 0.75

typedef struct {
	int rank;
    float value;
	float prev_value;
	float prev_prev_value;
} gridNode;

//Current Grid
gridNode* input_grid;
gridNode* output_grid;

//Setup the grid
void setupInputGrid(){
	int i;
	for (i=0;i<GRID_SIZE*GRID_SIZE;i++){
		// Allocate memory to each gridNode pointer that will contain the info relative to a node
		input_grid = (gridNode*) malloc(GRID_SIZE*GRID_SIZE*sizeof(gridNode));
	
		//Fill gridNodes
		if(i == (GRID_SIZE*GRID_SIZE/2)){
			//Center node (Set value to 1.0 to simulate a hit on the drum, It will be shifted to prev_value before the iteration)
			*(input_grid+i)->value = 1.0;
			*(input_grid+i)->prev_value = 0.0;
			*(input_grid+i)->prev_prev_value = 0.0;
		}else{
			*(input_grid+i)->value = 0.0;
			*(input_grid+i)->prev_value = 0.0;
			*(input_grid+i)->prev_prev_value = 0.0;
		}
	}
}

//512x512 worker function
__global__ void processGrid(gridNode* cud_input[GRID_SIZE][GRID_SIZE], gridNode* cud_output[GRID_SIZE][GRID_SIZE], int iterations){
	//int row = blockIdx.y * blockDim.y + threadIdx.y;
	//int col = blockIdx.x * blockDim.x + threadIdx.x;

}

void printMiddleValue(){
	//Print value at [GRID_SIZE/2,GRID_SIZE/2]
	float midValue = *(input_grid+(GRID_SIZE*GRID_SIZE/2))->value;
	printf("%f\n", midValue);
}

// Handler function for 512x512
int grid512x512_handler(int iterations) {
    gridNode* cud_input;
    gridNode* cud_output;
	output_grid = (gridNode*) malloc(GRID_SIZE*GRID_SIZE*sizeof(gridNode));
    int io_size = sizeof(input_grid);
	
    //Allocate mem on GPU
    cudaMalloc(&cud_input,io_size);
    cudaMalloc(&cud_output,io_size);
    
    //Copy the loaded image array to the GPU
    cudaMemcpy(cud_input, input_grid,io_size,cudaMemcpyHostToDevice);
      
    //Grid and block dims
    dim3 dimBlock(GRID_SIZE*sizeof(gridNode), 1, 1);
	dim3 dimGrid(GRID_SIZE, 1, 1);

    //Kernel
    GpuTimer timer;
    timer.Start();

    processGrid<<<dimGrid, dimBlock>>>(cud_input, cud_output, iterations);

    //Get back the processed grid
    cudaMemcpy(output_grid, cud_output,io_size,cudaMemcpyDeviceToHost);
    
    timer.Stop();
    float run_time = timer.Elapsed();

    //Finish up with the remaining work done (master does it, not slaves)
	//printMiddleValue();
    
    printf("\n Grid512x512 took: %f\n", run_time);
   
    //Cleanup
    cudaFree(cud_input);
    cudaFree (cud_output);

    return 0;
}

//MAIN
int main(int argc, char *argv[])
{
    //Usage
    if(argc<2)
    {
        printf("INVALID NUMBER OF ARGUMENTS Usage: ./grid 512 512 <iterations> \n");
        return 0;
    }
    
    //Get args
    int iterations = atoi(argv[1]);
	
	//Setup the input_grid
    setupInputGrid();
	
    //Call process
    grid512x512_handler(iterations);
    
    return 0;
}

