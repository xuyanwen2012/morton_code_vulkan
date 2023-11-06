#pragma once

#include <glm/glm.hpp>

constexpr glm::uint ExpandBits32(const glm::uint a) noexcept {
  glm::uint x = a & 0x000003FF;
  x = (x | x << 16) & 0x030000FF;
  x = (x | x << 8) & 0x0300F00F;
  x = (x | x << 4) & 0x030C30C3;
  x = (x | x << 2) & 0x09249249;
  return x;
}

constexpr uint32_t Encode32(const uint16_t x, const uint16_t y,
                            const uint16_t z) noexcept {
  return ExpandBits32(x) | (ExpandBits32(y) << 1) | (ExpandBits32(z) << 2);
}

constexpr uint32_t PointToCode32(const float x, const float y, const float z,
                                 const float min_coord,
                                 const float range) noexcept {
  constexpr uint32_t kCodeLen = 31;
  constexpr uint32_t bit_scale = 0xFFFFFFFFu >> (32 - (kCodeLen / 3));
  const auto x_coord =
      static_cast<uint32_t>(bit_scale * ((x - min_coord) / range));
  const auto y_coord =
      static_cast<uint32_t>(bit_scale * ((y - min_coord) / range));
  const auto z_coord =
      static_cast<uint32_t>(bit_scale * ((z - min_coord) / range));
  return Encode32(x_coord, y_coord, z_coord);
}

constexpr uint32_t PointToCode32(const glm::vec3 xyz, const float min_coord,
                                 const float range) noexcept {
  return PointToCode32(xyz.x, xyz.y, xyz.z, min_coord, range);
}