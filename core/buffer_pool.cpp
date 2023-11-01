#include "buffer_pool.hpp"

#include <algorithm>

namespace core {

BufferAllocation::BufferAllocation(MYBuffer &buffer, VkDeviceSize size,
                                   VkDeviceSize offset)
    : buffer{&buffer}, base_offset{offset}, size{size} {}

BufferBlock &BufferPool::request_buffer_block(const VkDeviceSize minimum_size,
                                              bool minimal) {

  // Find a block in the range of the blocks which can fit the minimum size
  auto it = minimal
                ? std::ranges::find_if(
                      buffer_blocks,
                      [&minimum_size](
                          const std::unique_ptr<BufferBlock> &buffer_block) {
                        return (buffer_block->get_size() == minimum_size) &&
                               buffer_block->can_allocate(minimum_size);
                      })
                : std::ranges::find_if(
                      buffer_blocks,
                      [&minimum_size](
                          const std::unique_ptr<BufferBlock> &buffer_block) {
                        return buffer_block->can_allocate(minimum_size);
                      });
}

} // namespace core