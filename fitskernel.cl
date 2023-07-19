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
      acc += G * data[y_idx * x_len + x_idx] * massMul * native_recip(x_dist * x_dist + y_dist * y_dist);
    }
  }

  result[y_pos * x_len + x_pos] = acc;
}