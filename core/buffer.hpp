#pragma once

#include <cstring>
#include <iostream>
#include <vector>

#include "../vma_usage.h"
#include "error.hpp"

namespace core {

extern VmaAllocator allocator;

// Unifed Shared Memory
class MYBuffer {
public:
  // Buffer() = default;
  MYBuffer() = delete;

  MYBuffer(const VkDeviceSize size) : size(size) { init(size); }

  MYBuffer(const MYBuffer &) = delete;

  MYBuffer(MYBuffer &&other)
      : alloc(other.alloc), memory{other.memory}, size{other.size},
        mapped_data{other.mapped_data} {

    // Reset other handles to avoid releasing on destruction
    other.alloc = VK_NULL_HANDLE;
    other.memory = VK_NULL_HANDLE;
    other.mapped_data = nullptr;
  }

  ~MYBuffer() {
    if (alloc != VK_NULL_HANDLE) {
      std::cout << "Destroying Buffer\n";
      vmaDestroyBuffer(allocator, buf, alloc);
    }
  }

  MYBuffer &operator=(const MYBuffer &) = delete;
  MYBuffer &operator=(MYBuffer &&) = delete;

  void init(const VkDeviceSize new_size) {
    size = new_size;

    // The default setting, Unified Shared Memory
    constexpr VmaAllocationCreateInfo alloc_create_info{
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                 VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT |
                 VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO,
    };

    const VkBufferCreateInfo buffer_create_info{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = new_size,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
    };

    const auto result =
        vmaCreateBuffer(allocator, &buffer_create_info, &alloc_create_info,
                        &buf, &alloc, &alloc_info);

    if (result != VK_SUCCESS) {
      throw VulkanException{result, "Cannot create Buffer"};
    }

    if constexpr (true) {
      std::cout << "alloc_info: " << std::endl;
      std::cout << "\tsize: " << alloc_info.size << std::endl;
      std::cout << "\toffset: " << alloc_info.offset << std::endl;
      std::cout << "\tmemoryType: " << alloc_info.memoryType << std::endl;
      std::cout << "\tmappedData: " << alloc_info.pMappedData << std::endl;
      std::cout << "\tdeviceMemory: " << alloc_info.deviceMemory << std::endl;
    }

    memory = alloc_info.deviceMemory;
    mapped_data = static_cast<std::byte *>(alloc_info.pMappedData);
  }

  const VkBuffer *get() const { return &buf; };
  VkBuffer *get_mut() { return &buf; };

  VmaAllocation get_allocation() const { return alloc; };
  VkDeviceMemory get_memory() const { return memory; }
  VkDeviceSize get_size() const { return size; };
  const std::byte *get_data() const { return mapped_data; }

  void update(const std::vector<std::byte> &data, size_t offset) {
    update(data.data(), data.size(), offset);
  }

  void update(const void *data, const size_t size, const size_t offset) {
    update(reinterpret_cast<const std::byte *>(data), size, offset);
  }

  void update(const std::byte *data, const size_t size, const size_t offset) {
    std::memcpy(mapped_data + offset, data, size);
  }

public:
  VmaAllocation alloc;
  VkBuffer buf;

  VmaAllocationInfo alloc_info;
  std::byte *mapped_data{nullptr};
  VkDeviceMemory memory{VK_NULL_HANDLE};
  VkDeviceSize size{0};
};

} // namespace core