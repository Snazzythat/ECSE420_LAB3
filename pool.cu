#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>

#define DIV 4

// Main pool function
__global__ void pool(unsigned char * d_out, unsigned char * d_in){
    
    int R, G, B;
    
    int curr = sizeof(unsigned char) * 4 * blockDim.x * 2 * (blockIdx.x * 2) + (sizeof(unsigned char) * 4 * threadIdx.x * 2);
    int down = sizeof(unsigned char) * 4 * blockDim.x * 2 * (blockIdx.x * 2 + 1) + (sizeof(unsigned char) * 4 *threadIdx.x * 2);
    int offs = sizeof(unsigned char) * 4 * blockDim.x * 2 * (blockIdx.x * 2) + (sizeof(unsigned char) * 4 * threadIdx.x * 2) + sizeof(unsigned char) * 4;
    int diag = sizeof(unsigned char) * 4 * blockDim.x * 2 * (blockIdx.x * 2 + 1) + (sizeof(unsigned char) * 4 *threadIdx.x * 2) + sizeof(unsigned char) * 4;
    
    
    int tempR;
    int tempR_a;
    int tempR_b;
    
    tempR_a = (d_in[curr]>d_in[down]) ? d_in[curr] : d_in[down];
    tempR_b = (d_in[offs]>d_in[diag]) ? d_in[offs] : d_in[diag];
    tempR = (tempR_a > tempR_b) ? tempR_a : tempR_b;
    
    int tempG;
    int tempG_a;
    int tempG_b;
    
    tempG_a = (d_in[curr + 1]>d_in[down + 1]) ? d_in[curr + 1] : d_in[down + 1];
    tempG_b = (d_in[offs + 1]>d_in[diag + 1]) ? d_in[offs + 1] : d_in[diag + 1];
    tempG = (tempG_a > tempG_b) ? tempG_a : tempG_b;
    
    int tempB;
    int tempB_a;
    int tempB_b;
    
    tempB_a = (d_in[curr + 2]>d_in[down + 2]) ? d_in[curr + 2] : d_in[down + 2];
    tempB_b = (d_in[offs + 2]>d_in[diag + 2]) ? d_in[offs + 2] : d_in[diag + 2];
    tempB = (tempB_a > tempB_b) ? tempB_a : tempB_b;

    
    int out_ind = sizeof(unsigned char) * 4 *blockDim.x * blockIdx.x + threadIdx.x * sizeof(unsigned char) * 4;
    
    d_out[out_ind] = tempR;
    d_out[out_ind + 1] = tempG;
    d_out[out_ind + 2] = tempB;
    d_out[out_ind + 3] = 255;
    
}


void pool_handler(char* input_file, char* output_file)
{
    
    unsigned loading_err;
    unsigned char *loaded_image, *output_image;
    unsigned input_image_width, input_image_height;
    unsigned char * cud_input;
    unsigned char * cud_output;
    
    //Loading the image via lodepng
    loading_err = lodepng_decode32_file(&loaded_image, &input_image_width, &input_image_height, input_file);
    if (loading_err)
        printf("error %u: %s\n", loading_err, lodepng_error_text(loading_err));
    
    int filtered_img_size = input_image_width * input_image_height * sizeof(unsigned char);
    unsigned blck_dim = input_image_width/2;
    unsigned grid_dim = input_image_height/2;
    
    output_image = (unsigned char *)malloc(filtered_img_size);
    
    
    // GPU input/output mem alloc
    int input_img_size = input_image_width * input_image_height * sizeof(unsigned char) * 4;
    cudaMalloc(&cud_input, input_img_size);
    cudaMalloc(&cud_output, filtered_img_size);
    
    // Send iput array to GPU
    cudaMemcpy(cud_input, loaded_image, input_img_size, cudaMemcpyHostToDevice);
    
    // Blocks per grid
    dim3 dimGrid(grid_dim, 1, 1);
    
    //Threads per block
    dim3 dimBlock(blck_dim , 1, 1);
    
    // Kernel
    pool<<<dimGrid, dimBlock>>>(cud_output, cud_input);
    
    // Get back the filtered data to the output array
    cudaMemcpy(output_image, cud_output, filtered_img_size, cudaMemcpyDeviceToHost);
    
    // Write back to PNG
    lodepng_encode32_file(output_file, output_image, blck_dim, grid_dim);
    
    // Cleanup
    cudaFree(cud_input);
    cudaFree(cud_output);
    free(loaded_image);
    free(output_image);

}

//MAIN
int main(int argc, char *argv[])
{
    //Usage
    if(argc<3)
    {
        printf("INVALID NUMBER OF ARGUMENTS Usage: ./pool <input PNG> <output PNG> \n");
        return 0;
    }
    
    //Get args
    char* in_fname = argv[1];
    char* out_fname = argv[2];
    
    //Call process
    pool_handler(in_fname, out_fname);
    
    return 0;
}
