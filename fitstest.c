#include "fitstest.h"

int main(int argc, char *argv[]) {
  int rc = 0, status = 0;
  fitsfile *input_file = NULL, *output_file = NULL;
  int hdutype, n_axes;
  long *axes_len = NULL, *coord = NULL;
  double *input_data = NULL, *results = NULL, zero = 0;
  int platform_index = -1, device_index = -1;
  if (argc == 1) {
    printf("Usage: [input file] [platform id] [device id]\n");
    return 0;
  }

  rc = fits_open_image(&input_file, argv[1], READONLY, &status);
  if (rc) {
    fits_report_error(stderr, status);
    return 0;
  }

  // Must be an image
  rc = fits_get_hdu_type(input_file, &hdutype, &status);
  if (rc) {
    fprintf(stderr, "Failed to get fits hdu type");
    fits_report_error(stderr, status);
    goto nope;
  }

  if (hdutype != IMAGE_HDU) {
    fprintf(stderr, "Input file must be an image\n");
    goto nope;
  }

  rc = fits_get_img_dim(input_file, &n_axes, &status);
  if (rc) {
    fprintf(stderr, "Could not get number of axes in image\n");
    fits_report_error(stderr, status);
    goto nope;
  }

  printf("Image has %d axes\n", n_axes);
  axes_len = malloc(sizeof(long) * n_axes);
  rc = fits_get_img_size(input_file, n_axes, axes_len, &status);
  if (rc) {
    fprintf(stderr, "Could not get image size\n");
    fits_report_error(stderr, status);
  }

  printf("Image is ");
  for(int i = 0; i < n_axes; i++) {
    printf("%ld", axes_len[i]); 
    if (i == n_axes - 1) printf("\n"); 
    else printf("x");
  }

  if (n_axes < 2) {
    printf("Need a 2D image\n");
    goto nope;
  }

  input_data = malloc(sizeof(double) * axes_len[0] * axes_len[1]);
  coord = malloc(sizeof(long) * n_axes);
  for (int i = 0; i < n_axes; i++) coord[i] = 1L;
  int anynul = 0;
  rc = fits_read_pix(input_file, TDOUBLE, coord, axes_len[0] * axes_len[1], 0, input_data, 0, &status);
  if (rc) {
    fprintf(stderr, "Could not read image data\n");
    fits_report_error(stderr, status);
    goto nope;
  }

  // verify data
  double sum = 0;
  for (int i = 0; i < axes_len[1]; i++)
    for (int j = 0; j < axes_len[0]; j++) 
      sum += input_data[i * axes_len[0] + j];

  if (sum == 0) {
    fprintf(stderr, "CFITSIO failed to read image data. Data is all zero\n");
    goto nope;
  } else {
    fprintf(stderr, "sum = %f\n", sum);
  }

  if (argc >= 4) {
    platform_index = atoi(argv[2]);
    device_index = atoi(argv[3]);
  }

  get_context_from_user(platform_index, device_index);
  if (context == NULL) {
    fprintf(stderr, "Could not get OpenCL context\n");
    goto nope;
  }

  // verify data
  sum = 0;
  for (int i = 0; i < axes_len[1]; i++)
    for (int j = 0; j < axes_len[0]; j++) 
      sum += input_data[i * axes_len[0] + j];

  if (sum == 0) {
    fprintf(stderr, "CFITSIO failed to read image data. Data is all zero\n");
    goto nope;
  } 

  results = calculate_potential(input_data, axes_len[0], axes_len[1]);
  rc = fits_create_file(&output_file, "out.fits", &status);
  if (rc) {
    fprintf(stderr, "Could not create output file\n");
    fits_report_error(stderr, status);
  }

  rc = fits_create_img(output_file, DOUBLE_IMG, 2, axes_len, &status);
  if (rc) {
    fprintf(stderr, "Could not create output image\n");
    fits_report_error(stderr, status);
  }

  fits_write_pix(output_file, TDOUBLE, coord, axes_len[0] * axes_len[1], results, &status); 
  if (rc) {
    fprintf(stderr, "Could not write pixels to output\n");
    fits_report_error(stderr, status);
  }

nope:
  free(axes_len);
  free(input_data);
  free(results);
  rc = fits_close_file(input_file, &status);
  printf("Input file closed. Status %d, rc %d\n", status, rc);
  if(rc) fits_report_error(stderr, status);
  rc = fits_close_file(output_file, &status);
  printf("Output file closed. Status %d, rc %d\n", status, rc);

  return 0;
}
