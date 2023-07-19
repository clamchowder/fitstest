#include <stdio.h>

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include "fitsio.h"

#define MAX_SOURCE_SIZE (0x100000)

cl_device_id selected_device_id;
cl_platform_id selected_platform_id; 
cl_context context; 

void get_context_from_user(int platform_index, int device_index); 
double *calculate_potential(double *data, long x_len, long y_len); 
double *omp_calculate_potential(double* data, long x_len, long y_len);
