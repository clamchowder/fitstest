#include <stdio.h>
#include "fitsio.h"

int main(int argc, char *argv[]) {
  int rc = 0, status = 0;
  fitsfile *input_file;
  if (argc == 1) {
    printf("Usage: [input file]\n");
    return 0;
  }

  rc = fits_open_file(&input_file, argv[1], READONLY, &status);
  if (!rc) {
    printf("Could not open file\n");
    return 0;
  }

  printf("File opened. Status %d, rc %d\n", status, rc);

  rc = fits_close_file(input_file, &status);
  printf("File closed. Status %d, rc %d\n", status, rc);

  return 0;
}
