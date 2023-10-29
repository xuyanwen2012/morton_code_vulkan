#include <array>
#include <iostream>
#include <vector>
#include <vulkan/vulkan_core.h>

#include "VkBootstrap.h"
#include "file_reader.hpp"
#include "vma_usage.h"

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

  void run(const std::vector<float> &input_data) {
    VkCheck(create_storage_buffer());
    VkCheck(write_data_to_buffer(input_data.data(), input_data.size()));
    VkCheck(create_descriptor_set(buffer));
    VkCheck(execute(input_data));
  }

  void teardown() { cleanup(); }

protected:
  /**
   * @brief Initialize vulkan device using vk-bootstrap
   *
   * @return int
   */
  [[nodiscard]] int device_initialization() {
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

    vkb::PhysicalDeviceSelector selector{instance};
    auto phys_ret =
        selector.defer_surface_initialization()
            .set_minimum_version(1, 1) // require a vulkan 1.1 capable device
            .require_separate_compute_queue()
            .select();

    if (!phys_ret) {
      std::cerr << "Failed to select Vulkan Physical Device. Error: "
                << phys_ret.error().message() << "\n";
      return -1;
    }
    std::cout << "selected GPU: " << phys_ret.value().properties.deviceName
              << '\n';

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
  CreateShaderModule(const std::vector<char> &code) {
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
        .vkGetDeviceProcAddr = &vkGetDeviceProcAddr};

    const VmaAllocatorCreateInfo allocator_create_info = {
        .physicalDevice = device.physical_device,
        .device = device.device,
        .pVulkanFunctions = &vulkan_functions,
        .instance = instance.instance};
    vmaCreateAllocator(&allocator_create_info, &allocator);
  }

  /**
   * @brief Create a descriptor set layout object
   *
   * @return int
   */
  [[nodiscard]] int create_descriptor_set_layout() {
    const std::array<VkDescriptorSetLayoutBinding, 1> binding{
        VkDescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
    };

    const VkDescriptorSetLayoutCreateInfo create_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = binding.data(),
    };

    if (disp.createDescriptorSetLayout(&create_info, nullptr,
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
    const auto compute_shader_code = readFile("shaders/square.spv");
    const auto compute_module = CreateShaderModule(compute_shader_code);

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
    constexpr VkDescriptorPoolSize pool_size{
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
    };

    // ReSharper disable once CppVariableCanBeMadeConstexpr
    const VkDescriptorPoolCreateInfo create_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_size,
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
  [[nodiscard]] int create_descriptor_set(const VkBuffer &buffer) {
    const VkDescriptorSetAllocateInfo alloc_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &descriptor_set_layout,
    };

    if (disp.allocateDescriptorSets(&alloc_info, &descriptor_set) !=
        VK_SUCCESS) {
      std::cout << "failed to allocate descriptor set\n";
      return -1;
    }

    const VkDescriptorBufferInfo buffer_info{
        .buffer = buffer,
        .offset = 0,
        .range = kN * sizeof(float),
    };

    const VkWriteDescriptorSet write{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = descriptor_set,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buffer_info,
    };

    disp.updateDescriptorSets(1, &write, 0, nullptr);

    return 0;
  }

  /**
   * @brief Create a storage buffer object
   *
   * @return int
   */
  [[nodiscard]] int create_storage_buffer() {
    constexpr VkBufferCreateInfo buffer_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = kN * sizeof(float),
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
    };

    constexpr VmaAllocationCreateInfo alloc_info{
        .usage = VMA_MEMORY_USAGE_CPU_TO_GPU,
        .requiredFlags = VMA_ALLOCATION_CREATE_MAPPED_BIT,
    };

    // VmaAllocation allocation;
    vmaCreateBuffer(allocator, &buffer_info, &alloc_info, &buffer, &allocation,
                    nullptr);

    return 0;
  }

  /**
   * @brief Write data to buffer
   *
   * @param h_data host data
   * @param n size of data
   * @return int
   */
  int write_data_to_buffer(const float *h_data, const size_t n) {
    assert(n * sizeof(float) == kN * sizeof(float));
    void *mapped_memory = nullptr;
    vmaMapMemory(allocator, allocation, &mapped_memory);
    memcpy(mapped_memory, h_data, n * sizeof(float));
    vmaUnmapMemory(allocator, allocation);
    return 0;
  }

  int execute(const std::vector<float> &input_data) {
    std::cout << "input data:\n";
    for (size_t i = 0; i < input_data.size(); ++i) {
      if (i % 64 == 0 && i != 0)
        std::cout << '\n';
      std::cout << input_data[i];
    }
    std::cout << '\n';

    // -------

    const VkCommandBufferAllocateInfo alloc_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };

    if (disp.allocateCommandBuffers(&alloc_info, &command_buffer) !=
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

    constexpr auto group_count_x =
        static_cast<uint32_t>(InputSize() / ComputeShaderProcessUnit());
    disp.cmdDispatch(command_buffer, group_count_x, 1, 1);

    disp.endCommandBuffer(command_buffer);

    // -------

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
    if (vkQueueWaitIdle(compute_queue) != VK_SUCCESS)
      throw std::runtime_error("failed to wait queue idle!");

    // copy data from GPU to CPU
    std::vector<float> output_data(kN);

    void *mapped_data;
    vmaMapMemory(allocator, allocation, &mapped_data);
    memcpy(output_data.data(), mapped_data, kN * sizeof(float));
    vmaUnmapMemory(allocator, allocation);
    // -------

    std::cout << "output data:\n";
    for (size_t i = 0; i < output_data.size(); ++i) {
      if (i % 64 == 0 && i != 0)
        std::cout << '\n';
      std::cout << output_data[i];
    }
    std::cout << '\n';
    return 0;
  }

  /**
   * @brief Destroy vulkan instance
   *
   */
  void cleanup() {
    vmaDestroyBuffer(allocator, buffer, allocation);

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

private:
  // struct {
  vkb::Instance instance;
  std::vector<const char *> enabled_extensions;
  //};

  // Buffer related
  // struct {
  VmaAllocation allocation;
  // potentially VkDeviceMemory and VkDeviceSize here, but was handled by VMA
  VkBuffer buffer;
  // uint8_t *mapped_data{nullptr};
  // bool persistent{false}; // Whether the buffer is persistently mapped or
  // not
  // bool mapped{false}; // Whether the buffer has been mapped with
  // vmaMapMemory
  //};

  // Device Related
  // struct {
  vkb::Device device;
  VmaAllocator allocator;
  vkb::DispatchTable disp;
  // queues (vector of queues)
  VkQueue compute_queue;
  // Potentially fence poll here

  // Command Related
  VkCommandPool command_pool;
  VkCommandBuffer command_buffer;
  //};

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

int main() {
  const std::vector h_data(kN, 2.0f);

  ComputeEngine engine;
  engine.init();
  engine.run(h_data);
  engine.teardown();

  std::cout << "Exiting normally" << std::endl;
  return EXIT_SUCCESS;
}
