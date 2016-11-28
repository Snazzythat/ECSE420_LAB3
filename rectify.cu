#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>
#include "omp.h"
#include <math.h>

//Rectify worker function
__global__ void rectify(unsigned char * cud_input, unsigned char * cud_output){
    
    int R,G,B,A;
    
    int idx = 4*blockDim.x*blockIdx.x + threadIdx.x*4;
    
    R = cud_input[idx+0];
    G = cud_input[idx+1];
    B = cud_input[idx+2];
    A = cud_input[idx+3];
    
    if(R<127)
    {
        R = 127;
    }
    if(G<127)
    {
        G = 127;
    }
    if(B<127)
    {
        B = 127;
    }
    
    cud_output[idx+0] = R;
    cud_output[idx+1] = G;
    cud_output[idx+2] = B;
    cud_output[idx+3] = A;
}

//Handler to prepare the necessary arrays and call the CUDA GPU fns
void rectification_handler(char* input_filename, char* output_filename)
{
    unsigned loading_err;
    unsigned char *loaded_image, *output_image;
    unsigned input_image_width, input_image_height;
    
    //Loading the image via lodepng
    loading_err = lodepng_decode32_file(&loaded_image, &input_image_width, &input_image_height, input_filename);
    if (loading_err)
        printf("error %u: %s\n", loading_err, lodepng_error_text(loading_err));
    
    int output_image_size = input_image_width * input_image_height * 4 * sizeof(unsigned char);
    
    output_image = (unsigned char *)malloc(output_image_size);
    
    unsigned char * cud_input;
    unsigned char * cud_output;
    
    output_image = (unsigned char *)malloc(output_image_size);
    
    // Mem allocation on gpu
    cudaMalloc(&cud_input, output_image_size);
    cudaMalloc(&cud_output, output_image_size);
    
    // Send the laaded image from lodepng to the gpu, provided the size
    cudaMemcpy(cud_input, loaded_image, output_image_size, cudaMemcpyHostToDevice);
    
    // Define numebr of threads in each blck
    dim3 dimBlock(input_image_width  * sizeof(unsigned char), 1, 1);
    
    // Define how many blcks a grid has
    dim3 dimGrid(input_image_height, 1, 1);
    
    //Kernel begin
    rectify<<<dimGrid, dimBlock>>>(cud_input, cud_output);
    
    //Wait till done, get the processed imaeg array back
    cudaMemcpy(output_image, cud_output, output_image_size, cudaMemcpyDeviceToHost);
    
    //Load back to PNG
    lodepng_encode32_file(output_filename, output_image, input_image_width, input_image_height);
    
    //Free all arrays
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
        printf("INVALID NUMBER OF ARGUMENTS Usage: ./rectify <input PNG> <output PNG> \n");
        return 0;
    }
    
    //Get args
    char* in_fname = argv[1];
    char* out_fname = argv[2];
    
    //Call process
    rectification_handler(in_fname, out_fname);
    
    return 0;
}



