#pragma once

#include "../vma_usage.h"

//  Creates a scratch buffer using VMA with pre-defined usage flags

class ScratchBuffer {
public:
  ScratchBuffer(const VkDeviceSize size) {
    const VkBufferCreateInfo buffer_info{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
    };

    VmaAllocationCreateInfo memory_info{
        .usage = VMA_MEMORY_USAGE_GPU_ONLY,
    };

    VmaAllocationInfo allocation_info{};

    // const auto result =
    //     vmaCreateBuffer(device.get_memory_allocator(), &buffer_info,
    //                     &memory_info, &buffer, &allocation,
    //                     &allocation_info);

    // if (result != VK_SUCCESS) {
    //   throw std::runtime_error("Failed to create scratch buffer");
    // }

    memory = allocation_info.deviceMemory;

    // VkBufferDeviceAddressInfoKHR buffer_device_address_info{};
    // buffer_device_address_info.sType =
    //     VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    // buffer_device_address_info.buffer = handle;
    // device_address = vkGetBufferDeviceAddressKHR(device.get_handle(),
    //                                              &buffer_device_address_info);
  }

  //   ~ScratchBuffer() {
  //     if (handle != VK_NULL_HANDLE && allocation != VK_NULL_HANDLE) {
  //       vmaDestroyBuffer(device.get_memory_allocator(), handle, allocation);
  //     }
  //   }

private:
  //   uint64_t device_address{0};

  VkBuffer buffer;
  VmaAllocation allocation;
  VkDeviceMemory memory;
  VkDeviceSize size;
};
