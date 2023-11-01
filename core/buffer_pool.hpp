#pragma once

#include <memory>

#include "buffer.hpp"

namespace core {

class BufferAllocation {
public:
  BufferAllocation() = default;

  BufferAllocation(const BufferAllocation &) = delete;
  BufferAllocation(MYBuffer &buffer, VkDeviceSize size, VkDeviceSize offset);

  BufferAllocation(BufferAllocation &&) = default;
  BufferAllocation &operator=(const BufferAllocation &) = delete;
  BufferAllocation &operator=(BufferAllocation &&) = default;

private:
  MYBuffer *buffer{nullptr};
  VkDeviceSize base_offset{0};
  VkDeviceSize size{0};
};

class BufferBlock {
public:
  BufferBlock(VkDevice &device, VkDeviceSize size, VkBufferUsageFlags usage,
              VmaMemoryUsage memory_usage);
  MYBuffer buffer;
  VkDeviceSize offset{0};

  bool can_allocate(VkDeviceSize size) const;
  BufferAllocation allocate(VkDeviceSize size);
  VkDeviceSize get_size() const;
};

class BufferPool {
public:
  BufferPool(VkDevice &device, VkDeviceSize block_size,
             VkBufferUsageFlags usage,
             VmaMemoryUsage memory_usage = VMA_MEMORY_USAGE_AUTO);

  BufferBlock &request_buffer_block(VkDeviceSize minimum_size,
                                    bool minimal = false);

  void reset();

private:
  VkDevice &device;

  /// List of blocks requested
  std::vector<std::unique_ptr<BufferBlock>> buffer_blocks;

  /// Minimum size of the blocks
  VkDeviceSize block_size{0};

  VkBufferUsageFlags usage{};

  VmaMemoryUsage memory_usage{};
};

} // namespace core