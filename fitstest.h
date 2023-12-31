#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include "fitsio.h"

#define MAX_SOURCE_SIZE (0x100000)

extern cl_device_id selected_device_id;
extern cl_platform_id selected_platform_id; 
extern cl_context context; 

void get_context_from_user(int platform_index, int device_index); 
double *calculate_potential_ocl(double *data, long x_len, long y_len, int fp32); 
double* calculate_potential_incremental_ocl(double* data, long x_len, long y_len, int fp32);
double *omp_calculate_potential(double* data, long x_len, long y_len);
double* omp_calculate_potential_avx(double* data, long x_len, long y_len);
