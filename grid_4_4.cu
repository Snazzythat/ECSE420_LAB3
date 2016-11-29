#include <stdio.h>
#include <stdlib.h>
#include "gputimer.h"


//Locator: Returns 1 if threads work on interior nodes, 2 if threads work on edges and 3 if threads work on corners
__device__ int get_locattion(int thread_id)
{
    int center = 1;
    int edge = 2;
    int corner = 3;
    int return_code = 0;
    
    if (thread_id == 5 || thread_id == 6 || thread_id == 9 || thread_id == 10)
    {
        return center;
    }
}

//Main CORNER node update function, called from GPU
__device__ void corner_update(int thread_id, double *u){
    double gain = 0.75;
    
    //UPPER LEFT CORNER
    if (thread_id == 0)
    {
        u[thread_id] = gain*u[4];
    }
    //UPPER RIGHT CORNER
    else if (thread_id == 3)
    {
        u[thread_id] = gain*u[2];
    }
    //LOWER LEFT CORNER
    else if (thread_id == 12)
    {
        u[thread_id] = gain*u[8];
    }
    //LOWER RIGHT CORNER
    else if (thread_id == 15)
    {
        u[thread_id] = gain*u[14];
    }
    
}

//Main EDGE node update function, called from GPU
__device__ void edge_update(int thread_id, double *u)
{
    double gain = 0.75;
    
    //UPPER EDGE
    if (thread_id == 1 || thread_id == 2)
    {
        u[thread_id] = gain*u[thread_id+4];
    }
    
    //LOWER EDGE
    else if (thread_id == 13 || thread_id == 14)
    {
        u[thread_id] = gain*u[thread_id-4];
    }
    
    //LEFT EDGE
    else if (thread_id == 4 || thread_id == 8)
    {
        u[thread_id] = gain*u[thread_id+1];
    }
    //RIGHT EDGE
    else if (thread_id == 7 || thread_id == 11)
    {
        u[thread_id] = gain*u[thread_id-1];
    }
}

//Main INTERIOR node update function, called from GPU
__device__ void center_update(double *u, double *u1, double *u2,int thread_id)
{
    double etha = 0.0002;
    double rho = 0.5;
    if (thread_id == 10 || thread_id == 6 || thread_id == 9 || thread_id == 5)
    {
        double center_update_val = (rho *(u1[thread_id-1] + u1[thread_id+1] + u1[thread_id-4] + u1[thread_id+4] - 4*u1[thread_id]) + 2*u1[thread_id] - (1-etha)*u2[thread_id]) / (etha+1);
        u[thread_id] = center_update_val;
    }
}

//Main update mechanism that calls the center, boundary and corner updates
__global__ void iteration_manager(double *cud_u, double *cud_u1, double *cud_u2, int iterations, double *results){
    
    int thread_id = threadIdx.x;
    
    int grid_siz = 16 * sizeof(double);
    
    // For each iteration --> update interior, edges, corners, record the value at (1,1)
    for(int i = 0; i < iterations; i++){
        
        //Corner updates, sync because edges depend on inner nodes
        center_update(cud_u, cud_u1, cud_u2,thread_id);
        __syncthreads();
        
        //Edge updates, sync because corners depend on edge values
        edge_update(thread_id, cud_u);
        __syncthreads();
        
        //Corner updates
        corner_update(thread_id, cud_u);
        __syncthreads();
        
        //Do the translation from U2 --> U1 and U1 --> U
        memcpy(cud_u2, cud_u1, grid_siz);
        memcpy(cud_u1, cud_u, grid_siz);
        
        //Record always the position (1,1)
        results[i] = cud_u[10];
    }
    
}

//Main handler
void grid_master_handler(int iterations)
{
    
    //Keep track of u1, u2 and present u on host CPU
    double u[16];
    double u1[16];
    double u2[16];
    double output[iterations];
    
    //Initialize u, u1, u2 and output to zeros
    for (int i = 0; i < 16; i++)
    {
        u[i] = 0;
        u1[i] = 0;
        u2[i] = 0;
        
        if (i < iterations)
        {
            output[i] = 0;
        }
    }
    
    //Inser perturbation at (1,1)
    u1[10] = 1;
    
    // declare GPU memory pointers
    double *cud_u;
    double *cud_u1;
    double *cud_u2;
    double *cud_output;
    
    //Allocate mem on GPU necessary for all the arrays and the output
    cudaMalloc(&cud_u, 16 * sizeof(double));
    cudaMalloc(&cud_u1, 16 * sizeof(double));
    cudaMalloc(&cud_u2, 16 * sizeof(double));
    cudaMalloc(&cud_output, iterations * sizeof(double));
    
    
    //Send the arrays to the GPU, use cudaMemcpyHostToDevice
    cudaMemcpy(cud_u, u, 16 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cud_u1, u1, 16 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cud_u2, u2, 16 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cud_output, output, iterations * sizeof(double), cudaMemcpyHostToDevice);
    
    GpuTimer timer;
    timer.Start();
    
    //Grid and block dims
    dim3 dimGrid(16, 1, 1);
    dim3 dimBlock(1, 1, 1);
    
    //Kernel FN call, calltimer too to start measuring execution time on GPU
    iteration_manager<<<dimBlock, dimGrid>>>(cud_u, cud_u1, cud_u2, iterations, cud_output);
    
    timer.Stop();
    float run_time = timer.Elapsed();
    printf("\n Grid 4x4 took: %f\n", run_time);
    
    
    //GPU is done, get back the results, use cudaMemcpyDeviceToHost
    cudaMemcpy(u, cud_u, 16 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(u1, cud_u1, 16 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(u2, cud_u2, 16 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(output, cud_output, iterations * sizeof(double), cudaMemcpyDeviceToHost);
    
    //Cleanup
    cudaFree(cud_u);
    cudaFree(cud_u1);
    cudaFree(cud_u2);
    cudaFree(cud_output);
    
    printf("%f", output[0]);
    
    for(int i = 1; i < iterations; i++)
    {
        printf(", %f", output[i]);
    }
}

//MAIN
int main(int argc, char *argv[])
{
    //Usage
    if(argc<2)
    {
        printf("INVALID NUMBER OF ARGUMENTS Usage: ./grid_4_4 <iterations> \n");
        return 0;
    }
    
    //Get args
    int iterations = atoi(argv[1]);
    
    //Call process
    grid_master_handler(iterations);
    
    return 0;
}

