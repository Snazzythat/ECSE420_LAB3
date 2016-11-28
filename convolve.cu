#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>
#include "wm.h"
#include "gputimer.h"

#define BLCK_SIZ 512


//Main convolution funct
__global__ void convolve(unsigned char * cud_out, unsigned char * cud_in, unsigned img_width, unsigned img_height, float weights[3][3]){
    
    unsigned int res;
    int wd = (threadIdx.x + blockIdx.x * blockDim.x)/(img_width-2) + 1;
    int hg = (threadIdx.x + blockIdx.x * blockDim.x)%(img_width-2) + 1;
    
    for (int m = 0; m < 3; m++)
    {
        res = 0;
        for (int q = 0; q < 3; q++)
        {
            for (int r = 0; r < 3; r++)
            {
                unsigned int input_val = cud_in[4*img_width*(wd+q-1) + 4*(hg+r-1) + m];
                res = res + (input_val * weights[q][r]);
            }
        }
        
        if (res < 0)
        {
            res = 0;
        }
        
        if (res > 255)
        {
            res = 255;
        }
        
        //Assigne back to output
        cud_out[(img_width-2)*(wd-1)*4 + (hg-1)*4 + m] = res;
    }
    unsigned int A = cud_in[4*img_width*wd + 4*hg + 3];
    
    cud_out[4*(img_width-2)*(wd-1) + 4*(hg-1) + 3] = A;
}

// Handler function for convolution
int convolution_handler(char* input_filename, char* output_filename) {
    
    unsigned loading_err;
    unsigned char *loaded_image, *output_image;
    unsigned char * cud_input;
    unsigned char * cud_output;
    unsigned input_image_width, input_image_height;
    
    
    //Loading the image via lodepng
    loading_err = lodepng_decode32_file(&loaded_image, &input_image_width, &input_image_height, input_filename);
    if (loading_err)
        printf("error %u: %s\n", loading_err, lodepng_error_text(loading_err));
    
    unsigned input_img_size = input_image_width*input_image_height * 4 * sizeof (unsigned char);
    unsigned output_img_size = (input_image_width-2)*(input_image_height-2);
    unsigned output_img_bytes = output_img_size * 4 * sizeof (unsigned char);
    
    output_image =(unsigned char *) malloc(output_img_bytes);
    
    float **loaded_weights;
    
    //Alloc mem on GPU
    cudaMalloc(&cud_input,input_img_size);
    cudaMalloc(&cud_output,output_img_bytes);
    cudaMalloc(&loaded_weights, sizeof(loaded_weights));
    
    //Copy the loaded image array to the GPU
    cudaMemcpy(cud_input, loaded_image,input_img_size,cudaMemcpyHostToDevice);
    
    //Copy the weights matrix as well
    cudaMemcpy(loaded_weights, w, sizeof(w), cudaMemcpyHostToDevice);
    
    //Grid and block dims
    dim3 dimGrid(BLCK_SIZ*2, 1, 1);
    dim3 dimBlock(((input_image_width-2)*(input_image_height-2)/(BLCK_SIZ*2)), 1, 1);

    //Kernel
    GpuTimer timer;
    timer.Start();

    convolve<<<dimBlock,dimGrid>>>(cud_output,cud_input,input_image_width,input_image_height,(float(*)[3])loaded_weights);

    //Get back the processed array
    cudaMemcpy(output_image, cud_output,output_img_bytes,cudaMemcpyDeviceToHost);
    
    timer.Stop();
    float run_time = timer.Elapsed();

    //Finish up with the remaining work done (master does it, not slaves)
    int leftovers = ((input_image_width-2)*(input_image_height-2))%(BLCK_SIZ*2);
    unsigned int res;
    int  wd = input_image_height - 2;
    int  hg = input_image_width - 2;
    
    for (int n = leftovers; n > 0 ; n--)
    {
        for (int m = 0; m < 3; m++)
        {
            res = 0;
            
            for (int q = 0; q < 3; q++)
            {
                for (int r = 0; r < 3; r++)
                {
                    res = res +  loaded_image[4 * input_image_width * (wd + q - 1) + 4 * (hg + r - 1) + m] * w[q][r];
                }
            }
            
            if (res > 255)
            {
                res = 255;
            }
            
            if (res < 0)
            {
                res = 0;
            }
            
            output_image[4 * (input_image_width - 2) * (wd - 1) + 4 * (hg - 1) + m] = res;
        }
        
        //After R,G,B, just put A since we dont touch it
        unsigned int A = loaded_image[4 * input_image_width * wd + 4 * hg + 3];
        output_image[ 4 * (input_image_width - 2) * (wd - 1) + 4 * (hg - 1) + 3] = A;
        
        if ((hg-1) == 0)
        {
            wd = wd - 1;
            hg= input_image_width - 2;
        }
        else if (hg > 0)
        {
            hg = hg - 1;
        }
    }
    
    printf("\n Convolution took: %f\n", run_time);

    //Back to PNG
    lodepng_encode32_file(output_filename, output_image, input_image_width-2, input_image_height-2);
    
    //Cleanup
    cudaFree(cud_input);
    cudaFree (cud_output);
    cudaFree (loaded_weights);
    free(loaded_image);
    free(output_image);
    return 0;
    
}

//MAIN
int main(int argc, char *argv[])
{
    //Usage
    if(argc<3)
    {
        printf("INVALID NUMBER OF ARGUMENTS Usage: ./convolve <input PNG> <output PNG> \n");
        return 0;
    }
    
    //Get args
    char* in_fname = argv[1];
    char* out_fname = argv[2];
    
    //Call process
    convolution_handler(in_fname, out_fname);
    
    return 0;
}

