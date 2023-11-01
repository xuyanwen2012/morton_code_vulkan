#pragma once

#include <iostream>

#include "../vma_usage.h"
#include "VkBootstrap.h"
#include "utils.hpp"

namespace core {

// Need a global allocator for VMA
extern VmaAllocator allocator;

// Base Engine does The Setup. Single device, single queue
class BaseEngine {
public:
  BaseEngine() {
    vk_check(device_initialization());
    vk_check(get_queues());
    vma_initialization();
  }

  ~BaseEngine() {
    if (allocator != VK_NULL_HANDLE) {
      vmaDestroyAllocator(allocator);
    }
    destroy_device(device);
    destroy_instance(instance);
  }

private:
  [[nodiscard]] int device_initialization() {
    // Vulkan instance creation (1/3)
    vkb::InstanceBuilder instance_builder;
    auto inst_ret = instance_builder.set_app_name("Example Vulkan Application")
                        .request_validation_layers()
                        .use_default_debug_messenger()
                        .require_api_version(1, 3, 0) // for SPIR-V 1.3
                        .build();
    if (!inst_ret) {
      std::cerr << "Failed to create Vulkan instance. Error: "
                << inst_ret.error().message() << "\n";
      return -1;
    }

    instance = inst_ret.value();

    // Vulkan pick physical device (2/3)
    vkb::PhysicalDeviceSelector selector{instance};
    auto phys_ret =
        selector
            .defer_surface_initialization()
            // .set_minimum_version(1, 2)
            .prefer_gpu_device_type(vkb::PreferredDeviceType::integrated)
            .allow_any_gpu_device_type(false)
            .select();

    if (!phys_ret) {
      std::cerr << "Failed to select Vulkan Physical Device. Error: "
                << phys_ret.error().message() << "\n";
      return -1;
    }
    std::cout << "selected GPU: " << phys_ret.value().properties.deviceName
              << '\n';

    // Vulkan logical device creation (3/3)
    vkb::DeviceBuilder device_builder{phys_ret.value()};
    auto dev_ret = device_builder.build();
    if (!dev_ret) {
      std::cerr << "Failed to create Vulkan device. Error: "
                << dev_ret.error().message() << "\n";
      return -1;
    }

    device = dev_ret.value();
    disp = device.make_table();
    return 0;
  }

  [[nodiscard]] int get_queues() {
    auto q_ret = device.get_queue(vkb::QueueType::compute);
    if (!q_ret.has_value()) {
      std::cout << "failed to get compute queue: " << q_ret.error().message()
                << "\n";
      return -1;
    }
    compute_queue = q_ret.value();
    return 0;
  }

  void vma_initialization() {
    if (allocator != VK_NULL_HANDLE) {
      return;
    }

    constexpr VmaVulkanFunctions vulkan_functions{
        .vkGetInstanceProcAddr = &vkGetInstanceProcAddr,
        .vkGetDeviceProcAddr = &vkGetDeviceProcAddr,
    };

    const VmaAllocatorCreateInfo allocator_create_info{
        .physicalDevice = device.physical_device,
        .device = device.device,
        .pVulkanFunctions = &vulkan_functions,
        .instance = instance.instance,
    };

    vmaCreateAllocator(&allocator_create_info, &allocator);
  }

protected:
  vkb::Instance instance;
  vkb::Device device;
  vkb::DispatchTable disp;
  VkQueue compute_queue; // queues (vector of queues)
};

} // namespace core