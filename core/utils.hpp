#pragma once

#include <cstdlib>

inline void vk_check(const int result) {
  if (result != 0) {
    exit(1);
  }
}
