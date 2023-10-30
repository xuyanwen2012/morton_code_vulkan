#include <algorithm>
#include <array>
#include <cstring>
#include <glm/fwd.hpp>
#include <iostream>
#include <random>
#include <ranges>
#include <vector>

#include "VkBootstrap.h"
#include "vma_usage.h"
#include <glm/glm.hpp>
#include <vulkan/vulkan_core.h>

#include "file_reader.hpp"
#include "moton.hpp"

constexpr auto kN = 1024;

[[nodiscard]] constexpr uint32_t InputSize() { return kN; }
[[nodiscard]] constexpr uint32_t ComputeShaderProcessUnit() { return 256; }

inline void VkCheck(const int result) {
  if (result != 0) {
    exit(1);
  }
}
// ScratchBuffer

class ComputeEngine {
public:
  void init() {
    VkCheck(device_initialization());
    VkCheck(get_queues());
    vma_initialization();

    VkCheck(create_descriptor_set_layout());
    VkCheck(create_compute_pipeline());
    VkCheck(create_descriptor_pool());
    VkCheck(create_command_pool());
  }

  void run(const std::vector<glm::vec3> &input_data) {
    VkCheck(create_storage_buffer());
    VkCheck(create_descriptor_set());

    VkCheck(write_data_to_buffer(input_data.data(), input_data.size()));

    VkCheck(execute_sync());
  }

  void teardown() { cleanup(); }

protected:
  /**
   * @brief Initialize vulkan device using vk-bootstrap
   *
   * @return int
   */
  [[nodiscard]] int device_initialization() {
    // Vulkan instance creation (1/3)
    vkb::InstanceBuilder instance_builder;
    auto inst_ret = instance_builder.set_app_name("Example Vulkan Application")
                        .request_validation_layers()
                        .use_default_debug_messenger()
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
        selector.defer_surface_initialization()
            .set_minimum_version(1, 2)
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

  /**
   * @brief Create a Shader Module object from SPIR-V code
   *
   * @param code SPIR-V code
   * @return VkShaderModule shader module handle
   */
  [[nodiscard]] VkShaderModule
  create_shader_module(const std::vector<char> &code) {
    const VkShaderModuleCreateInfo create_info{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code.size(),
        .pCode = reinterpret_cast<const uint32_t *>(code.data()),
    };

    VkShaderModule shader_module;
    if (disp.createShaderModule(&create_info, nullptr, &shader_module) !=
        VK_SUCCESS) {
      return VK_NULL_HANDLE;
    }

    return shader_module;
  }

  /**
   * @brief Get the queues object from device
   *
   * @return int 0 if success, -1 if failed
   */
  [[nodiscard]] int get_queues() {
    auto cq = device.get_queue(vkb::QueueType::compute);
    if (!cq.has_value()) {
      std::cout << "failed to get graphics queue: " << cq.error().message()
                << "\n";
      return -1;
    }
    compute_queue = cq.value();
    return 0;
  }

  /**
   * @brief Initialize Vulkan Memory Allocator
   *
   */
  void vma_initialization() {
    constexpr VmaVulkanFunctions vulkan_functions = {
        .vkGetInstanceProcAddr = &vkGetInstanceProcAddr,
        .vkGetDeviceProcAddr = &vkGetDeviceProcAddr,
    };

    const VmaAllocatorCreateInfo allocator_create_info = {
        .physicalDevice = device.physical_device,
        .device = device.device,
        .pVulkanFunctions = &vulkan_functions,
        .instance = instance.instance,
    };

    vmaCreateAllocator(&allocator_create_info, &allocator);
  }

  /**
   * @brief Create a descriptor set layout object
   *
   * @return int
   */
  [[nodiscard]] int create_descriptor_set_layout() {
    // First thing to do
    const std::array<VkDescriptorSetLayoutBinding, 2> binding{
        // input
        VkDescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        // output
        VkDescriptorSetLayoutBinding{
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
    };

    const VkDescriptorSetLayoutCreateInfo layout_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 2,
        .pBindings = binding.data(),
    };

    if (disp.createDescriptorSetLayout(&layout_info, nullptr,
                                       &descriptor_set_layout) != VK_SUCCESS) {
      std::cout << "failed to create descriptor set layout\n";
      return -1;
    }

    return 0;
  }

  /**
   * @brief Create a command pool object
   *
   * @return int
   */
  [[nodiscard]] int create_command_pool() {
    const VkCommandPoolCreateInfo pool_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex =
            device.get_queue_index(vkb::QueueType::compute).value(),
    };

    if (disp.createCommandPool(&pool_info, nullptr, &command_pool) !=
        VK_SUCCESS) {
      std::cout << "failed to create command pool\n";
      return -1;
    }
    return 0;
  }

  /**
   * @brief Create a compute pipeline object
   *
   * @return int
   */
  [[nodiscard]] int create_compute_pipeline() {
    // Load & Create Shader Modules (1/3)
    const auto compute_shader_code = readFile("shaders/debug.spv");
    const auto compute_module = create_shader_module(compute_shader_code);

    if (compute_module == VK_NULL_HANDLE) {
      std::cout << "failed to create shader module\n";
      return -1;
    }

    const VkPipelineShaderStageCreateInfo shader_stage_create_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = compute_module,
        .pName = "main",
    };

    // Create a Pipeline Layout (2/3)
    const VkPipelineLayoutCreateInfo layout_create_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &descriptor_set_layout,
        .pushConstantRangeCount = 0,
        .pPushConstantRanges = nullptr,
    };
    if (disp.createPipelineLayout(&layout_create_info, nullptr,
                                  &compute_pipeline_layout) != VK_SUCCESS) {
      std::cout << "failed to create pipeline layout\n";
      return -1;
    }

    // Pipeline itself (3/3)
    const VkComputePipelineCreateInfo pipeline_create_info{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = shader_stage_create_info,
        .layout = compute_pipeline_layout,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1,
    };
    if (disp.createComputePipelines(VK_NULL_HANDLE, 1, &pipeline_create_info,
                                    nullptr, &compute_pipeline) != VK_SUCCESS) {
      std::cout << "failed to create compute pipeline\n";
      return -1;
    }

    disp.destroyShaderModule(compute_module, nullptr);
    return 0;
  }

  /**
   * @brief Create a descriptor pool object
   *
   * @return int
   */
  [[nodiscard]] int create_descriptor_pool() {
    constexpr std::array<VkDescriptorPoolSize, 2> pool_sizes{
        VkDescriptorPoolSize{
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
        },
        VkDescriptorPoolSize{
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1, // maybe 2?
        },
    };

    // ReSharper disable once CppVariableCanBeMadeConstexpr
    const VkDescriptorPoolCreateInfo create_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 2,
        .poolSizeCount = 2,
        .pPoolSizes = pool_sizes.data(),
    };

    if (disp.createDescriptorPool(&create_info, nullptr, &descriptor_pool) !=
        VK_SUCCESS) {
      std::cout << "failed to create descriptor pool\n";
      return -1;
    }
    return 0;
  }

  /**
   * @brief Create a descriptor set object
   *
   * @param buffer
   * @return int
   */
  [[nodiscard]] int create_descriptor_set() {
    const VkDescriptorSetAllocateInfo set_alloc_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &descriptor_set_layout,
    };

    if (disp.allocateDescriptorSets(&set_alloc_info, &descriptor_set) !=
        VK_SUCCESS) {
      std::cout << "failed to allocate descriptor set\n";
      return -1;
    }

    const VkDescriptorBufferInfo in_buffer_info{
        .buffer = buffers[0],
        .offset = 0,
        .range = kN * sizeof(glm::vec3),
    };

    const VkDescriptorBufferInfo out_buffer_info{
        .buffer = buffers[1], // why not [1]?
        .offset = 0,
        .range = kN * sizeof(glm::uint),
    };

    const std::array<VkWriteDescriptorSet, 2> write{
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &in_buffer_info,
        },
        VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 1,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &out_buffer_info,
        },
    };

    disp.updateDescriptorSets(2, write.data(), 0, nullptr);

    return 0;
  }

  /**
   * @brief Create a storage buffer object
   *
   * @return int
   */
  [[nodiscard]] int create_storage_buffer() {
    // Checkout
    // https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/usage_patterns.html
    //  It will then prefer a memory type that is both DEVICE_LOCAL and
    //  HOST_VISIBLE (integrated memory or BAR)
    constexpr VmaAllocationCreateInfo alloc_create_info{
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                 VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT |
                 VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO,
    };

    constexpr std::array<VkBufferCreateInfo, 2> buffer_create_info{
        VkBufferCreateInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = kN * sizeof(glm::vec3),
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
        },
        VkBufferCreateInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = kN * sizeof(glm::uint),
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
        },
    };

    for (auto i = 0; i < 2; ++i) {
      vmaCreateBuffer(allocator, &buffer_create_info[i], &alloc_create_info,
                      &buffers[i], &allocations[i], &alloc_info[i]);
      std::cout << "alloc_info: " << i << std::endl;
      std::cout << "size: " << alloc_info[i].size << std::endl;
      std::cout << "offset: " << alloc_info[i].offset << std::endl;
      std::cout << "memoryType: " << alloc_info[i].memoryType << std::endl;
      std::cout << "mappedData: " << alloc_info[i].pMappedData << std::endl;
      std::cout << "deviceMemory: " << alloc_info[i].deviceMemory << std::endl;
    }

    // Print all alloc_info info
    if (allocations[0] == VK_NULL_HANDLE || allocations[1] == VK_NULL_HANDLE) {
      std::cout << "failed to allocate buffer\n";
      return -1;
    }

    // Check if the memory is host visible
    VkMemoryPropertyFlags memPropFlags;
    vmaGetAllocationMemoryProperties(allocator, allocations[0], &memPropFlags);

    if (memPropFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
      std::cout << "host visible" << std::endl;
    } else {
      std::cout << "not host visible" << std::endl;
      return -1;
    }
    return 0;
  }

  /**
   * @brief Write data to buffer
   *
   * @param h_data host data
   * @param n size of data
   * @return int
   */
  int write_data_to_buffer(const glm::vec3 *h_data, const size_t n) {
    assert(n * sizeof(glm::vec3) == kN * sizeof(glm::vec3));
    memcpy(alloc_info[0].pMappedData, h_data, sizeof(glm::vec3) * n);
    return 0;
  }

  [[nodiscard]] int execute_sync() {
    const VkCommandBufferAllocateInfo cmd_buf_alloc_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };

    if (disp.allocateCommandBuffers(&cmd_buf_alloc_info, &command_buffer) !=
        VK_SUCCESS) {
      std::cout << "failed to allocate command buffers\n";
      return -1;
    }

    // ------- RECORD COMMAND BUFFER --------
    constexpr VkCommandBufferBeginInfo begin_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    };
    disp.beginCommandBuffer(command_buffer, &begin_info);

    disp.cmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                         compute_pipeline);
    disp.cmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               compute_pipeline_layout, 0, 1, &descriptor_set,
                               0, nullptr);

    // equvalent to CUDA's number of blocks
    constexpr auto group_count_x =
        static_cast<uint32_t>(InputSize() / ComputeShaderProcessUnit());
    disp.cmdDispatch(command_buffer, group_count_x, 1, 1);

    disp.endCommandBuffer(command_buffer);

    // ------- SUBMIT COMMAND BUFFER --------
    const VkSubmitInfo submit_info{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = 0,
        .commandBufferCount = 1,
        .pCommandBuffers = &command_buffer,
        .signalSemaphoreCount = 0,
    };

    if (vkQueueSubmit(compute_queue, 1, &submit_info, VK_NULL_HANDLE) !=
        VK_SUCCESS) {
      std::cout << "failed to submit queue\n";
      return -1;
    }

    // wait the calculation to finish
    if (vkQueueWaitIdle(compute_queue) != VK_SUCCESS) {
      throw std::runtime_error("failed to wait queue idle!");
    }

    return 0;
  }

  /**
   * @brief Destroy vulkan instance
   *
   */
  void cleanup() {
    for (int i = 0; i < 2; ++i) {
      vmaDestroyBuffer(allocator, buffers[i], allocations[i]);
    }
    // vmaDestroyBuffer(allocator, buffer, allocation);

    if (allocator != VK_NULL_HANDLE) {
      vmaDestroyAllocator(allocator);
    }

    disp.destroyDescriptorPool(descriptor_pool, nullptr);
    disp.destroyCommandPool(command_pool, nullptr);
    disp.destroyDescriptorSetLayout(descriptor_set_layout, nullptr);
    disp.destroyPipeline(compute_pipeline, nullptr);
    disp.destroyPipelineLayout(compute_pipeline_layout, nullptr);
    destroy_device(device);
    destroy_instance(instance);
  }

public:
  vkb::Instance instance;

  // Buffer related
  // VmaAllocation allocation;
  std::array<VmaAllocation, 2> allocations;
  std::array<VkBuffer, 2> buffers;
  std::array<VmaAllocationInfo, 2> alloc_info;

  // Device Related
  vkb::Device device;
  VmaAllocator allocator;
  vkb::DispatchTable disp;
  VkQueue compute_queue; // queues (vector of queues)
  // Potentially fence poll here

  // Command Related
  VkCommandPool command_pool;
  VkCommandBuffer command_buffer;

  // Compute Related
  VkDescriptorSetLayout descriptor_set_layout;
  VkDescriptorPool descriptor_pool;
  VkDescriptorSet descriptor_set;

  // Pipeline Related
  VkPipelineLayout compute_pipeline_layout;
  VkPipeline compute_pipeline;
  // std::vector<VkShaderModule*> shader_modules;

  // The shader resources that this pipeline layout uses, indexed by their name
  // A map of each set and the resources it owns used by the pipeline layout
  // The different descriptor set layouts for this pipeline layout
};

std::ostream &operator<<(std::ostream &os, const glm::vec3 &vec) {
  os << "(" << vec.x << ", " << vec.y << ", " << vec.z << ")";
  return os;
}

int main() {
  std::default_random_engine gen(114514);
  std::uniform_real_distribution<float> dis(0.0f, 1024.0f);

  std::vector<glm::vec3> h_data2(kN);
  std::ranges::generate(
      h_data2, [&]() { return glm::vec3(dis(gen), dis(gen), dis(gen)); });

  std::cout << "Input:\n";
  for (size_t i = 0; i < 10; ++i) {
    std::cout << h_data2[i] << '\n';
  }

  ComputeEngine engine;
  engine.init();

  engine.run(h_data2);

  if (engine.alloc_info[1].pMappedData != nullptr) {
    // -------
    auto output_data =
        reinterpret_cast<glm::uint *>(engine.alloc_info[1].pMappedData);

    std::cout << "Output:\n";
    for (size_t i = 0; i < 10; ++i) {

      const auto code = Debug(h_data2[i]);

      std::cout << i << ":\t" << h_data2[i] << "\t" << output_data[i] << '\t'
                << code << '\n';
    }
  }

  engine.teardown();

  std::cout << "Done\n";
  return EXIT_SUCCESS;
}
