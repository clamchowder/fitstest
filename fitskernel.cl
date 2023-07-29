// Assumes local size = 256
// utterly unoptimized
__kernel void calculate_potential(__global double *data, int x_len, int y_len, __global double *result) {
  double G = 6.67430E-8;         // gravitational constant but with grams not kg
  double massMul = 2.3*1.67E-80; 
  double pxDistance = 1.76E16;    // 1.76E32 sq cm
  int localid = get_local_id(0);
  int x_pos = get_global_id(0) % x_len;
  int y_pos = get_global_id(0) / x_len;

  if (x_pos >= x_len || y_pos >= y_len) return;

  // compute gravitational potential for given position
  double acc = 0;
  for (int x_idx = 0; x_idx < x_len; x_idx++) {
    for (int y_idx = 0; y_idx < y_len; y_idx++) {
      double x_dist = (x_pos - x_idx) * pxDistance;
      double y_dist = (y_pos - y_idx) * pxDistance;
      if (y_idx == y_pos && x_idx == x_pos) continue;
      acc += G * data[y_idx * x_len + x_idx] * massMul * native_rsqrt(x_dist * x_dist + y_dist * y_dist);
    }
  }

  result[y_pos * x_len + x_pos] = acc;
}

__kernel void calculate_potential_fp32(__global float *data, int x_len, int y_len, __global float *result) {
  float G = 6.67430E-8;         // gravitational constant but with grams not kg
  float massMul = 2.3*1.67E-30; // shift from -80 into fp32 range
  float pxDistance = 1.76E16;    // 1.76E32 sq cm
  int localid = get_local_id(0);
  int x_pos = get_global_id(0) % x_len;
  int y_pos = get_global_id(0) / x_len;

  if (x_pos >= x_len || y_pos >= y_len) return;

  // compute gravitational potential for given position
  float acc = 0;
  for (int x_idx = 0; x_idx < x_len; x_idx++) {
    for (int y_idx = 0; y_idx < y_len; y_idx++) {
      float x_dist = (x_pos - x_idx) * pxDistance;
      float y_dist = (y_pos - y_idx) * pxDistance;
      if (y_idx == y_pos && x_idx == x_pos) continue;
      acc += G * data[y_idx * x_len + x_idx] * massMul * native_rsqrt(x_dist * x_dist + y_dist * y_dist);
    }
  }

  result[y_pos * x_len + x_pos] = acc;
}

// Calculates results for a part of the image, hopefully small enough to avoid TDRs
// Also exposes a bit more parallelism. Uses LDS, assumes 256 items in wg, and row length is divisble by that
// data = data array
// x_len = image column count
// y_len = image row count
// y_offset = column to start at
// y_count = how many columns to go through. y_offset = 0 and y_count = y_len would go through the whole image
// result = output
__kernel void calculate_potential_partial(__global double *data, int x_len, int y_len, int y_offset, int y_count, __global double *result) {
  double G = 6.67430E-8;         // gravitational constant but with grams not kg
  double massMul = 2.3*1.67E-80; 
  double pxDistance = 1.76E16;    // 1.76E32 sq cm
  int localSize = get_local_size(0);
  int localId = get_local_id(0);
  int x_pos = get_group_id(0) % x_len; // wg processes pixel
  int y_pos = get_global_id(0) / x_len + y_offset;

  __local double partial_acc[256];

  // mask off anything that is definitely out of range
  if (x_pos >= x_len || y_pos >= y_len) return;

  // compute gravitational potential for given position
  double acc = 0;
  double posMass = data[y_pos * x_len + x_pos];
  for (int y_idx = 0; y_idx < y_len; y_idx++) {
    // advance by double * wg len
    for (int x_idx = localId; x_idx < x_len; x_idx += localSize) {
      if (y_idx == y_pos && x_idx == x_pos) continue;
      double x_dist = (x_pos - x_idx) * pxDistance;
      double y_dist = (y_pos - y_idx) * pxDistance;
      acc += G * data[y_idx * x_len + x_idx] * massMul * native_rsqrt(x_dist * x_dist + y_dist * y_dist);
    }
  }

  partial_acc[localId] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);

  // reduction. wg size better be even or else
  for (int doIMatter = localSize / 2; doIMatter > 1; doIMatter /= 2) {
    if (localId < doIMatter) {
      partial_acc[localId] += partial_acc[localId * 2];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }

  result[y_pos * x_len + x_pos] = partial_acc[0] + partial_acc[1];
}