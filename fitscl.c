#include "fitstest.h"

#include <immintrin.h>
#include <math.h>

cl_device_id selected_device_id;     
cl_platform_id selected_platform_id; 
cl_context context;                  

/// <summary>
/// populate global variables for opencl device id and platform id
/// </summary>
/// <param name="platform_index">platform index. if -1, prompt user</param>
/// <param name="device_index">device index. if -1. prompt user</param>
void get_context_from_user(int platform_index, int device_index) {
    int i = 0;
    int selected_platform_index = 0, selected_device_index = 0;

    // Get platform and device information
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;

    cl_int ret = clGetPlatformIDs(0, NULL, &ret_num_platforms);
    cl_platform_id* platforms = NULL;
    cl_device_id* devices = NULL;
    context = NULL;
    platforms = (cl_platform_id*)malloc(ret_num_platforms * sizeof(cl_platform_id));

    ret = clGetPlatformIDs(ret_num_platforms, platforms, NULL);
    fprintf(stderr, "clGetPlatformIDs returned %d. %d platforms\n", ret, ret_num_platforms);

    for (i = 0; i < ret_num_platforms; i++)
    {
        size_t platform_name_len;
        char* platform_name = NULL;
        if (CL_SUCCESS != clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, NULL, &platform_name_len)) {
            fprintf(stderr, "Failed to get platform info for platform %d\n", i);
            continue;
        }

        platform_name = (char*)malloc(platform_name_len + 1);
        platform_name[platform_name_len] = 0;

        if (CL_SUCCESS != clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, platform_name_len, platform_name, NULL)) {
            fprintf(stderr, "Failed to get platform name for platform %d\n", i);
            free(platform_name);
            continue;
        }

        fprintf(stderr, "Platform %d: %s\n", i, platform_name);
        free(platform_name);
    }

    selected_platform_index = platform_index;
    if (selected_platform_index == -1)
    {
        printf("Enter platform #:");
        scanf("%d", &selected_platform_index);
    }

    if (selected_platform_index > ret_num_platforms - 1)
    {
        fprintf(stderr, "platform index out of range\n");
        goto get_context_from_user_end;
    }

    selected_platform_id = platforms[selected_platform_index];

    if (CL_SUCCESS != clGetDeviceIDs(selected_platform_id, CL_DEVICE_TYPE_ALL, 0, NULL, &ret_num_devices)) {
        fprintf(stderr, "Failed to enumerate device ids for platform");
        return;
    }

    devices = (cl_device_id*)malloc(ret_num_devices * sizeof(cl_device_id));
    if (CL_SUCCESS != clGetDeviceIDs(selected_platform_id, CL_DEVICE_TYPE_ALL, ret_num_devices, devices, NULL)) {
        fprintf(stderr, "Failed to get device ids for platform");
        free(devices);
        return;
    }

    fprintf(stderr, "clGetDeviceIDs returned %d devices\n", ret_num_devices);

    for (i = 0; i < ret_num_devices; i++)
    {
        size_t device_name_len;
        char* device_name = NULL;
        if (CL_SUCCESS != clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 0, NULL, &device_name_len)) {
            fprintf(stderr, "Failed to get name length for device %d\n", i);
            continue;
        }

        //fprintf(stderr, "debug: device name length: %d\n", device_name_len);
        device_name = (char*)malloc(device_name_len + 1);
        device_name[device_name_len] = 0;

        if (CL_SUCCESS != clGetDeviceInfo(devices[i], CL_DEVICE_NAME, device_name_len, device_name, &device_name_len)) {
            fprintf(stderr, "Failed to get name for device %d\n", i);
            free(device_name);
            continue;
        }

        fprintf(stderr, "Device %d: %s\n", i, device_name);
        free(device_name);
    }

    selected_device_index = device_index;
    if (selected_device_index == -1)
    {
        fprintf(stderr, "Enter device #:");
        scanf("%d", &selected_device_index);
    }


    if (selected_device_index > ret_num_devices - 1)
    {
        fprintf(stderr, "Device index out of range\n");
        goto get_context_from_user_end;
    }

    selected_device_id = devices[selected_device_index];

    // Create an OpenCL context
    context = clCreateContext(NULL, 1, &selected_device_id, NULL, NULL, &ret);
    fprintf(stderr, "clCreateContext returned %d\n", ret);

get_context_from_user_end:
    free(platforms);
    free(devices);
}

cl_program build_program(cl_context context, const char* fname)
{
    cl_int ret;
    FILE* fp = NULL;
    char* source_str;
    size_t source_size;
    fp = fopen(fname, "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel %s.\n", fname);
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t*)&source_size, &ret);
    ret = clBuildProgram(program, 1, &selected_device_id, NULL, NULL, NULL);
    //fprintf(stderr, "clBuildProgram %s returned %d\n", fname, ret);
    if (ret == -11)
    {
        size_t log_size;
        fprintf(stderr, "OpenCL kernel build error\n");
        clGetProgramBuildInfo(program, selected_device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, selected_device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "%s\n", log);
        free(log);
    }

    free(source_str);
    return program;
}

double G = 6.67430E-8;         // gravitational constant but with grams not kg
double massMul = 1; //2.3 * 1.67E-80;
double pxDistance = 1; //1.76E16;    // 1.76E32 sq cm

double *omp_calculate_potential(double* data, long x_len, long y_len)
{
    size_t buffer_size = sizeof(double) * x_len * y_len;
    double* results = malloc(buffer_size);
    int y_pos, x_pos;

    fprintf(stderr, "Using OpenMP and C code\n");

#pragma omp parallel for
    for (y_pos = 0; y_pos < y_len; y_pos++) {
        for (x_pos = 0; x_pos < x_len; x_pos++) {
            double m = data[y_pos * x_len + x_pos] * massMul;
            double acc = 0;
            for (int x_idx = 0; x_idx < x_len; x_idx++) {
                for (int y_idx = 0; y_idx < y_len; y_idx++) {
                    double x_dist = (x_pos - x_idx) * pxDistance;
                    double y_dist = (y_pos - y_idx) * pxDistance;
                    if (x_dist == 0 && y_dist == 0) continue;
                    acc += G * m * data[y_idx * x_len + x_idx] * massMul / (x_dist * x_dist + y_dist * y_dist);
                }
            }

            results[y_pos * x_len + x_pos] = acc;
        }
    }
    return results;
}

// attempt at writing an avx version
double* omp_calculate_potential_avx(double* data, long x_len, long y_len)
{
#ifdef _MSC_VER
    size_t buffer_size = sizeof(double) * x_len * y_len;
    double* results = malloc(buffer_size);
    int y_pos, x_pos;

    uint64_t pixel_offsets_arr[4] = {0, 1, 2, 3};
    uint32_t pixel_offsets_arr_32[4] = { 0, 1, 2, 3 };
    __m256i pixel_offsets_64b = _mm256_loadu_epi64(pixel_offsets_arr);
    __m128i pixel_offsets_32b = _mm_loadu_epi32(pixel_offsets_arr_32);
    __m256i x_len_vec = _mm256_set1_epi64x(x_len);
    __m256d distance_vec = _mm256_set1_pd(pxDistance);
    __m256d g_vec = _mm256_set1_pd(G);
    __m256d massMul_vec = _mm256_set1_pd(massMul);

    size_t padded_buffer_size = sizeof(double) * (x_len + 4) * (y_len + 4);
    long padded_x_len = x_len + 4;
    double* padded_data = malloc(padded_buffer_size);
    if (padded_data == NULL)
    {
        fprintf(stderr, "Could not allocate memory for padded buffer\n");
        return results;
    }

    fprintf(stderr, "Using AVX2 intrinsics\n");
    
    // pad the right/bottom edges with four extra black pixels
    for (y_pos = 0; y_pos < y_len + 4; y_pos++)
        for (x_pos = 0; x_pos < x_len + 4; x_pos++)
        {
            padded_data[y_pos * padded_x_len + x_pos] = (y_pos < y_len && x_pos < x_len) ? data[y_pos * x_len + x_pos] : 0;
        }

    start_timing();

#pragma omp parallel for
    for (y_pos = 0; y_pos < y_len; y_pos++) {
        // handle 4 px at a time (256-bit)
        for (x_pos = 0; x_pos < x_len; x_pos += 4) {
            __m256d acc_pixel_mass = _mm256_loadu_pd(padded_data + (y_pos * padded_x_len + x_pos));
            acc_pixel_mass = _mm256_mul_pd(acc_pixel_mass, massMul_vec);
            __m256d acc = _mm256_setzero_pd();

            // acc pixel position in FP64 vector
            __m128i acc_pixel_offsets_32 = _mm_add_epi32(pixel_offsets_32b, _mm_set1_epi32(x_pos));
            __m256d acc_pixel_offsets_f64 = _mm256_cvtepi32_pd(acc_pixel_offsets_32);
            for (int y_idx = 0; y_idx < y_len; y_idx++) {
                __m256d y_dist_squared = _mm256_mul_pd(_mm256_set1_pd(y_pos - y_idx), distance_vec);
                y_dist_squared = _mm256_mul_pd(y_dist_squared, y_dist_squared);
                __m256i y_pos_vec = _mm256_set1_epi64x(y_idx);
                for (int x_idx = 0; x_idx < x_len; x_idx += 4) {
                    if (y_idx == y_pos && x_idx == x_pos) { // should be equal bc both move by 4
                        // handle acc for four pixels
                        double temp_acc[4];
                        temp_acc[0] = 0;
                        temp_acc[1] = 0;
                        temp_acc[2] = 0;
                        temp_acc[3] = 0;
                        double y_dist = y_idx - y_pos;
                        for (int mini_x_pos = x_pos; mini_x_pos < x_pos + 4; mini_x_pos++) {
                            double m = data[y_pos * x_len + mini_x_pos] * massMul;
                            for (int mini_x_idx = x_idx; mini_x_idx < x_idx + 4; mini_x_idx++) {
                                if (mini_x_pos == mini_x_idx) continue;
                                double x_dist = mini_x_idx - mini_x_pos;
                                temp_acc[mini_x_pos - x_pos] += G * m * data[y_idx * x_len + x_idx] * massMul / sqrt(x_dist * x_dist + y_dist * y_dist);
                            }
                        }

                        __m256d temp_acc_vec = _mm256_loadu_pd(temp_acc);
                        acc = _mm256_add_pd(acc, temp_acc_vec);
                    }
                    else {
                        __m256d current_pixel_mass = _mm256_loadu_pd(padded_data + (y_idx * padded_x_len + x_idx));
                        current_pixel_mass = _mm256_mul_pd(current_pixel_mass, massMul_vec);

                        __m128i current_pixel_offsets_32 = _mm_add_epi32(pixel_offsets_32b, _mm_set1_epi32(x_idx));

                        // generate x distance values as doubles
                        __m256d current_pixel_offsets_f64 = _mm256_cvtepi32_pd(current_pixel_offsets_32);
                        __m256d x_dist = _mm256_mul_pd(_mm256_sub_pd(acc_pixel_offsets_f64, current_pixel_offsets_f64), distance_vec);

                        // compute (G*m1*m2/d^2)
                        __m256d numerator = _mm256_mul_pd(current_pixel_mass, acc_pixel_mass);
                        __m256d denominator = _mm256_sqrt_pd(_mm256_fmadd_pd(x_dist, x_dist, y_dist_squared)); // multiply first two args, add that to third arg
                        __m256d divideResult = _mm256_div_pd(numerator, denominator);
                        acc = _mm256_fmadd_pd(g_vec, divideResult, acc);
                    }
                }
            }

            _mm256_storeu_pd(results + (y_pos * x_len + x_pos), acc);
        }
    }

    printf("%d ms\n", end_timing());
    free(padded_data);
    return results;
#endif
}

// compute gravitational potential of column density
// data = ptr to data array, in row major order
// x_len = length of horizontal dimension in pixels
// y_len = length of vertical dimension in pixels
// fp32 = if true, use fp32 for calculations
// returns ptr to results, which must be freed
double* calculate_potential(double* data, long x_len, long y_len, int fp32) {
    cl_int ret;
    size_t global_size = x_len * y_len;
    size_t local_size = 256;

    // Nvidia does not like taking 64-bit integers as kernel arguments. Their runtime will start hurting itself.
    // 32-bit is fine here because images will not be too big anyway
    cl_int x_len_int = (cl_int)x_len;
    cl_int y_len_int = (cl_int)y_len;
    cl_program program = build_program(context, "fitskernel.cl");
    cl_command_queue command_queue = clCreateCommandQueue(context, selected_device_id, 0, &ret);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "Failed to create command queue: %d\n", ret);
    }

    cl_kernel kernel = clCreateKernel(program, fp32 ? "calculate_potential_fp32" : "calculate_potential", &ret);
    size_t buffer_size = sizeof(double) * x_len * y_len;
    size_t sp_buffer_size = sizeof(float) * x_len * y_len;

    double* results = malloc(buffer_size);
    cl_mem results_mem = NULL, data_mem = NULL;
    float* sp_input = NULL, *sp_results = NULL;

    if (fp32)
    {
        sp_input = malloc(sp_buffer_size);
        for (int y_pos = 0; y_pos < y_len; y_pos++)
            for (int x_pos = 0; x_pos < x_len; x_pos++)
                sp_input[y_pos * x_len + x_pos] = (float)data[y_pos * x_len + x_pos];

        sp_results = malloc(sp_buffer_size);

        data_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, sp_buffer_size, NULL, &ret);
        results_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sp_buffer_size, NULL, &ret);
        ret = clEnqueueWriteBuffer(command_queue, data_mem, CL_FALSE, 0, sp_buffer_size, sp_input, 0, NULL, NULL);
        ret = clEnqueueWriteBuffer(command_queue, results_mem, CL_FALSE, 0, sp_buffer_size, sp_results, 0, NULL, NULL);
    }
    else
    {
        data_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, buffer_size, NULL, &ret);
        results_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, &ret);
        ret = clEnqueueWriteBuffer(command_queue, data_mem, CL_FALSE, 0, buffer_size, data, 0, NULL, NULL);
        ret = clEnqueueWriteBuffer(command_queue, results_mem, CL_FALSE, 0, buffer_size, results, 0, NULL, NULL);
    }

    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&data_mem);
    clSetKernelArg(kernel, 1, sizeof(cl_int), (void *)&x_len_int);
    clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&y_len_int);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&results_mem);
    clFinish(command_queue);

    if (global_size % local_size != 0) global_size += (local_size - (global_size % local_size));

    printf("Submitting kernel to GPU\n");
    start_timing();
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "Failed to submit kernel: %d\n", ret);
        goto calculate_potential_end;
    }

    ret = clFinish(command_queue);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "Failed to finish command queue: %d\n", ret);
        goto calculate_potential_end;
    }
  
    printf("%d ms\n", end_timing());

    ret = clEnqueueReadBuffer(command_queue, results_mem, CL_TRUE, 0, fp32 ? sp_buffer_size : buffer_size, fp32 ? sp_results : results, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "Failed to copy results from GPU: %d\n", ret);
    }

    printf("Finished copying results back from GPU\n");

    if (fp32)
    {
        for (int y_pos = 0; y_pos < y_len; y_pos++)
            for (int x_pos = 0; x_pos < x_len; x_pos++)
                results[y_pos * x_len + x_pos] = sp_results[y_pos * x_len + x_pos];
        free(sp_input);
        free(sp_results);
    }

calculate_potential_end:
    clFlush(command_queue);
    clFinish(command_queue);
    clReleaseMemObject(data_mem);
    clReleaseMemObject(results_mem);
    return results;
}
