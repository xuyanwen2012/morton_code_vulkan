
uint expand_bit(uint a);
// uint encode(uint3 ijk);
uint encode(uint i, uint j, uint k);

struct MyPushConstant {
  uint n;
  float min_coord;
  float range;
};

uint expand_bit(uint a) {
  uint x = a & 0x000003FF;
  x = (x | (x << 16)) & 0x030000FF;
  x = (x | (x << 8)) & 0x0300F00F;
  x = (x | (x << 4)) & 0x030C30C3;
  x = (x | (x << 2)) & 0x09249249;
  return x;
}

// uint encode(uint3 ijk) {
// return expand_bit(ijk.x) | expand_bit(ijk.y) << 1 | expand_bit(ijk.z) << 2;
// }

uint encode(uint i, uint j, uint k) {
  return expand_bit(i) | expand_bit(j) << 1 | expand_bit(k) << 2;
}

__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void
foo(__global float4 *in_xyz, __global uint *out,
    const struct MyPushConstant push_constant) {
  const uint kCodeLen = 31;

  float x, y, z;
  uint i, j, k;

  uint bit_scale;
  float bit_scale_f;

  uint n = push_constant.n;
  float min_coord = push_constant.min_coord;
  float range = push_constant.range;

  // uint n = ;
  // float min_coord = 0.0f;
  // float range = 1024.0f;

  uint index = get_global_id(0);
  if (index >= n)
    return;

  // printf("index: %d\n", index);

  x = in_xyz[index].x;
  y = in_xyz[index].y;
  z = in_xyz[index].z;
  bit_scale = 0xFFFFFFFFu >> (32 - (kCodeLen / 3)); // 1023
  bit_scale_f = convert_float(bit_scale);           // 1023

  i = convert_uint(bit_scale_f * ((x - min_coord) / range));
  j = convert_uint(bit_scale_f * ((y - min_coord) / range));
  k = convert_uint(bit_scale_f * ((z - min_coord) / range));

  out[index] = encode(i, j, k);
}