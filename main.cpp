#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <ranges>
#include <vector>

#include "file_reader.hpp"
#include "moton.hpp"
#include "vma_usage.h"
#include <glm/glm.hpp>

#include "core/base_engine.hpp"

constexpr auto kN = 1024;

using InputT = glm::vec4;
using OutputT = glm::uint;

[[nodiscard]] constexpr uint32_t InputSize() { return kN; }
[[nodiscard]] constexpr uint32_t ComputeShaderProcessUnit() { return 256; }

struct MyPushConsts {
  uint32_t n;
  float min_coord;
  float range;
};

namespace core {
VmaAllocator allocator;

class ComputeEngine : public core::BaseEngine {
public:
  ComputeEngine() : BaseEngine() {
    vk_check(create_descriptor_set_layout());
    vk_check(create_descriptor_pool());
    vk_check(create_storage_buffer());
    vk_check(create_descriptor_set());
    vk_check(create_compute_pipeline());

    vk_check(create_command_pool());
  }

  ~ComputeEngine() {
    for (int i = 0; i < 2; ++i) {
      vmaDestroyBuffer(allocator, buffers[i], allocations[i]);
    }

    disp.destroyDescriptorPool(descriptor_pool, nullptr);
    disp.destroyCommandPool(command_pool, nullptr);
    disp.destroyDescriptorSetLayout(descriptor_set_layout, nullptr);
    disp.destroyPipeline(compute_pipeline, nullptr);
    disp.destroyPipelineLayout(compute_pipeline_layout, nullptr);
  }

  void run(const std::vector<InputT> &input_data) {
    vk_check(write_data_to_buffer(input_data.data(), input_data.size()));
    vk_check(execute_sync());
  }

protected:
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
    const auto compute_shader_code = readFile("shaders/morton.spv");
    const auto compute_module = create_shader_module(compute_shader_code);

    if (compute_module == VK_NULL_HANDLE) {
      std::cout << "failed to create shader module\n";
      return -1;
    }

    // Set constant IDs
    constexpr std::size_t num_of_entries = 3u;
    const std::array<VkSpecializationMapEntry, num_of_entries> spec_map{
        VkSpecializationMapEntry{
            .constantID = 0,
            .offset = 0,
            .size = sizeof(uint32_t),
        },
        VkSpecializationMapEntry{
            .constantID = 1,
            .offset = 4,
            .size = sizeof(uint32_t),
        },
        VkSpecializationMapEntry{
            .constantID = 2,
            .offset = 8,
            .size = sizeof(uint32_t),
        },
    };

    constexpr std::array<uint32_t, num_of_entries> spec_map_content{
        ComputeShaderProcessUnit(), 1, 1};

    const VkSpecializationInfo spec_info{
        .mapEntryCount = num_of_entries,
        .pMapEntries = spec_map.data(),
        .dataSize = sizeof(uint32_t) * num_of_entries,
        .pData = spec_map_content.data(),
    };

    // Shader stage create info
    const VkPipelineShaderStageCreateInfo shader_stage_create_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = compute_module,
        .pName = "foo",
        .pSpecializationInfo = &spec_info,
    };

    // pushconstant,name,region_offset,offset,0,size,12
    constexpr VkPushConstantRange push_const{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = 16 + sizeof(MyPushConsts), // 16 is for the first 3 uint32_t, 12
                                           // is for the second struct
    };

    // Create a Pipeline Layout (2/3)
    const VkPipelineLayoutCreateInfo layout_create_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &descriptor_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_const,
    };

    if (disp.createPipelineLayout(&layout_create_info, nullptr,
                                  &compute_pipeline_layout) != VK_SUCCESS) {
      std::cout << "failed to create pipeline layout\n";
      return -1;
    }

    // Pipeline itself (3/3)
    // Pipeline create info
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
        .range = InputSize() * sizeof(InputT),
    };

    const VkDescriptorBufferInfo out_buffer_info{
        .buffer = buffers[1], // why not [1]?
        .offset = 0,
        .range = InputSize() * sizeof(OutputT),
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
            .size = InputSize() * sizeof(InputT),
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
        },
        VkBufferCreateInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = InputSize() * sizeof(OutputT),
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
      std::cout << "\tsize: " << alloc_info[i].size << std::endl;
      std::cout << "\toffset: " << alloc_info[i].offset << std::endl;
      std::cout << "\tmemoryType: " << alloc_info[i].memoryType << std::endl;
      std::cout << "\tmappedData: " << alloc_info[i].pMappedData << std::endl;
      std::cout << "\tdeviceMemory: " << alloc_info[i].deviceMemory
                << std::endl;
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
      // std::cout << "host visible" << std::endl;
    } else {
      // std::cout << "not host visible" << std::endl;
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
  int write_data_to_buffer(const InputT *h_data, const size_t n) {
    memcpy(alloc_info[0].pMappedData, h_data, sizeof(InputT) * n);
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

    constexpr uint32_t default_push[3]{0, 0, 0};
    disp.cmdPushConstants(command_buffer, compute_pipeline_layout,
                          VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, &default_push[0]);

    constexpr MyPushConsts push_const{InputSize(), 0.0f, 1024.0f};
    disp.cmdPushConstants(command_buffer, compute_pipeline_layout,
                          VK_SHADER_STAGE_COMPUTE_BIT, 16, sizeof(MyPushConsts),
                          &push_const);

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

public:
  // Buffer related
  std::array<VmaAllocation, 2> allocations;
  std::array<VkBuffer, 2> buffers;
  std::array<VmaAllocationInfo, 2> alloc_info; // to access the mapped memory

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
};

}; // namespace core

std::ostream &operator<<(std::ostream &os, const glm::vec4 &vec) {
  os << "(" << vec.x << ", " << vec.y << ", " << vec.z << ")";
  return os;
}

void ShowAvailableExtensions() {
  uint32_t extensionCount = 0;
  vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

  std::vector<VkExtensionProperties> extensions(extensionCount);
  vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount,
                                         extensions.data());

  std::cout << "Available Vulkan Extensions:" << std::endl;
  for (const auto &extension : extensions) {
    std::cout << "\t" << extension.extensionName << std::endl;
  }
}

int main() {
  // Prepare data
  std::default_random_engine gen(114514);
  std::uniform_real_distribution<float> dis(0.0f, 1024.0f);

  std::vector<InputT> h_data2(InputSize());
  std::ranges::generate(
      h_data2, [&]() { return glm::vec4(dis(gen), dis(gen), dis(gen), 0.0f); });

  // Init
  core::ComputeEngine engine{};

  // Execute
  engine.run(h_data2);

  if (engine.alloc_info[1].pMappedData != nullptr) {
    // -------
    auto output_data =
        reinterpret_cast<OutputT *>(engine.alloc_info[1].pMappedData);

    std::cout << "Output:\n";
    for (size_t i = 0; i < 10; ++i) {

      const auto code = PointToCode(h_data2[i]);

      std::cout << i << ":\t" << h_data2[i] << "\t" << output_data[i];
      std::cout << '\t' << code;
      std::cout << '\n';
    }
  }

  std::cout << "Done\n";
  return EXIT_SUCCESS;
}
