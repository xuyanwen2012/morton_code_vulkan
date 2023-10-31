void kernel foo(global float *in, global float *out) {
  uint index = get_global_id(0);
  out[index] = in[index] * 2.0f;
}
