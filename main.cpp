#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

#include "file_reader.hpp"
#include "moton.hpp"

#include "vk_mem_alloc.h"

#include <glm/glm.hpp>

#include "core/base_engine.hpp"
#include "core/error.hpp"

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

// Unifed Shared Memory
class Buffer {
public:
  Buffer() = default;
  // MYBuffer() = delete;

  Buffer(const VkDeviceSize size) : size(size) { init(size); }

  // MYBuffer(const MYBuffer &) = delete;

  // MYBuffer(MYBuffer &&other)
  //     : alloc(other.alloc), memory{other.memory}, size{other.size},
  //       mapped_data{other.mapped_data} {

  //   // Reset other handles to avoid releasing on destruction
  //   other.alloc = VK_NULL_HANDLE;
  //   other.memory = VK_NULL_HANDLE;
  //   other.mapped_data = nullptr;
  // }

  ~Buffer() {
    if (alloc != VK_NULL_HANDLE) {
      std::cout << "Destroying Buffer\n";
      vmaDestroyBuffer(allocator, buf, alloc);
    }
  }

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

// Each GPU function is a kernel, it needs its own pipeline
//
// vkCreateComputePipelines
//   - VkComputePipelineCreateInfo
//     - VkPipelineShaderStageCreateInfo
//       - VkSpecializationInfo
//       - Shader Module
//   - vkCreatePipelineLayout
//     - VkPipelineLayoutCreateInfo
//       - Descriptor Set Layout
//       - Push Constants
//
struct ComputeKernel {
  VkPipeline pipeline;
  VkPipelineLayout pipelineLayout;
};

class ComputeEngine : public core::BaseEngine {
public:
  ComputeEngine() : BaseEngine() {
    create_descriptor_set_layout();
    create_descriptor_pool();

    // usm_buffers[0] = std::make_unique<Buffer>(InputSize() * sizeof(InputT));
    // usm_buffers[1] = std::make_unique<Buffer>(InputSize() * sizeof(OutputT));
    // vk_check(create_storage_buffer());

    // usm_buffers.push_back(std::move(MYBuffer(InputSize() * sizeof(InputT))));
    // usm_buffers.push_back(std::move(MYBuffer(InputSize() *
    // sizeof(OutputT))));

    usm_buffers[0].init(InputSize() * sizeof(InputT));
    usm_buffers[1].init(InputSize() * sizeof(OutputT));

    create_descriptor_set();
    create_compute_pipeline();

    create_command_pool();
  }

  ~ComputeEngine() {
    disp.destroyDescriptorPool(descriptor_pool, nullptr);
    disp.destroyCommandPool(command_pool, nullptr);
    disp.destroyDescriptorSetLayout(descriptor_set_layout, nullptr);
    disp.destroyPipeline(compute_pipeline, nullptr);
    disp.destroyPipelineLayout(compute_pipeline_layout, nullptr);
  }

  void run(const std::vector<InputT> &input_data) {
    write_data_to_buffer(input_data.data(), input_data.size());
    execute_sync();
  }

protected:
  [[nodiscard]] VkShaderModule
  create_shader_module(const std::vector<char> &code) {
    const VkShaderModuleCreateInfo create_info{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code.size(),
        .pCode = reinterpret_cast<const uint32_t *>(code.data()),
    };

    VkShaderModule shader_module;

    if (const auto result =
            disp.createShaderModule(&create_info, nullptr, &shader_module);
        result != VK_SUCCESS) {
      throw VulkanException{result, "Cannot create shader module"};
    }

    return shader_module;
  }

  void create_descriptor_set_layout() {
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

    if (const auto result = disp.createDescriptorSetLayout(
            &layout_info, nullptr, &descriptor_set_layout);
        result != VK_SUCCESS) {
      throw VulkanException{result, "Cannot create descriptor set layout"};
    }
  }

  void create_command_pool() {
    const VkCommandPoolCreateInfo pool_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex =
            device.get_queue_index(vkb::QueueType::compute).value(),
    };

    if (const auto result =
            disp.createCommandPool(&pool_info, nullptr, &command_pool);
        result != VK_SUCCESS) {
      throw VulkanException{result, "Cannot create command pool"};
    }
  }

  void create_compute_pipeline() {
    // Load & Create Shader Modules (1/3)
    const auto compute_shader_code = readFile("shaders/morton.spv");
    const auto compute_module = create_shader_module(compute_shader_code);

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

    if (const auto result = disp.createPipelineLayout(
            &layout_create_info, nullptr, &compute_pipeline_layout);
        result != VK_SUCCESS) {
      throw VulkanException{result, "Cannot create pipeline layout"};
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

    if (const auto result = disp.createComputePipelines(
            VK_NULL_HANDLE, 1, &pipeline_create_info, nullptr,
            &compute_pipeline);
        result != VK_SUCCESS) {
      throw VulkanException{result, "Cannot create compute pipeline"};
    }

    disp.destroyShaderModule(compute_module, nullptr);
  }

  void create_descriptor_pool() {
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

    if (const auto result =
            disp.createDescriptorPool(&create_info, nullptr, &descriptor_pool);
        result != VK_SUCCESS) {
      throw VulkanException{result, "Cannot create descriptor pool"};
    }
  }

  void create_descriptor_set() {
    const VkDescriptorSetAllocateInfo set_alloc_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &descriptor_set_layout,
    };

    if (const auto result =
            disp.allocateDescriptorSets(&set_alloc_info, &descriptor_set);
        result != VK_SUCCESS) {
      throw VulkanException{result, "Cannot allocate descriptor set"};
    }

    // Maybe combine into struct buffer
    const VkDescriptorBufferInfo in_buffer_info{
        .buffer = usm_buffers[0].buf,
        // .buffer = buffers[0],
        .offset = 0,
        .range = InputSize() * sizeof(InputT),
    };

    const VkDescriptorBufferInfo out_buffer_info{
        .buffer = usm_buffers[1].buf, // why not [1]?
        // .buffer = buffers[1],
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
  }

  // /**
  //  * @brief Create a storage buffer object
  //  *
  //  * @return int
  //  */
  // [[nodiscard]] int create_storage_buffer() {
  // // Checkout
  // //
  // https: //
  // gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/usage_patterns.html
  //   //  It will then prefer a memory type that is both DEVICE_LOCAL and
  //   //  HOST_VISIBLE (integrated memory or BAR)
  //   constexpr VmaAllocationCreateInfo alloc_create_info{
  //       .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
  //                VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT
  //                | VMA_ALLOCATION_CREATE_MAPPED_BIT,
  //       .usage = VMA_MEMORY_USAGE_AUTO,
  //   };

  //   constexpr std::array<VkBufferCreateInfo, 2> buffer_create_info{
  //       VkBufferCreateInfo{
  //           .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
  //           .size = InputSize() * sizeof(InputT),
  //           .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
  //                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
  //           .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
  //           .queueFamilyIndexCount = 0,
  //           .pQueueFamilyIndices = nullptr,
  //       },
  //       VkBufferCreateInfo{
  //           .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
  //           .size = InputSize() * sizeof(OutputT),
  //           .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
  //                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
  //           .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
  //           .queueFamilyIndexCount = 0,
  //           .pQueueFamilyIndices = nullptr,
  //       },
  //   };

  //   for (auto i = 0; i < 2; ++i) {
  //     vmaCreateBuffer(allocator, &buffer_create_info[i], &alloc_create_info,
  //                     &buffers[i], &allocations[i], &alloc_info[i]);
  //     std::cout << "alloc_info: " << i << std::endl;
  //     std::cout << "\tsize: " << alloc_info[i].size << std::endl;
  //     std::cout << "\toffset: " << alloc_info[i].offset << std::endl;
  //     std::cout << "\tmemoryType: " << alloc_info[i].memoryType << std::endl;
  //     std::cout << "\tmappedData: " << alloc_info[i].pMappedData <<
  //     std::endl; std::cout << "\tdeviceMemory: " <<
  //     alloc_info[i].deviceMemory
  //               << std::endl;
  //   }

  //   // Print all alloc_info info
  //   if (allocations[0] == VK_NULL_HANDLE || allocations[1] == VK_NULL_HANDLE)
  //   {
  //     std::cout << "failed to allocate buffer\n";
  //     return -1;
  //   }

  //   // Check if the memory is host visible
  //   VkMemoryPropertyFlags memPropFlags;
  //   vmaGetAllocationMemoryProperties(allocator, allocations[0],
  //   &memPropFlags);

  //   if (memPropFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
  //     // std::cout << "host visible" << std::endl;
  //   } else {
  //     // std::cout << "not host visible" << std::endl;
  //     return -1;
  //   }
  //   return 0;
  // }

  void write_data_to_buffer(const InputT *h_data, const size_t n) {
    usm_buffers[0].update(h_data, sizeof(InputT) * n, 0);
  }

  void execute_sync() {
    const VkCommandBufferAllocateInfo cmd_buf_alloc_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };

    if (const auto result =
            disp.allocateCommandBuffers(&cmd_buf_alloc_info, &command_buffer);
        result != VK_SUCCESS) {
      throw VulkanException{result, "Cannot allocate command buffer"};
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

    if (const auto result =
            disp.queueSubmit(compute_queue, 1, &submit_info, VK_NULL_HANDLE);
        result != VK_SUCCESS) {
      throw VulkanException{result, "Cannot submit queue"};
    }

    // wait the calculation to finish
    if (const auto result = disp.queueWaitIdle(compute_queue);
        result != VK_SUCCESS) {
      throw VulkanException{result, "Cannot wait queue idle"};
    }
  }

public:
  // Buffer related
  // std::array<VmaAllocation, 2> allocations;
  // std::array<VkBuffer, 2> buffers;
  // std::array<VmaAllocationInfo, 2> alloc_info; // to access the mapped memory

  std::array<Buffer, 2> usm_buffers;

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

std::ostream &operator<<(std::ostream &os, const glm::vec3 &vec) {
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

  // -------
  // auto output_data =
  // reinterpret_cast<OutputT *>(engine.alloc_info[1].pMappedData);

  auto output_data =
      reinterpret_cast<const OutputT *>(engine.usm_buffers[1].get_data());

  std::cout << "Output:\n";
  for (size_t i = 0; i < 10; ++i) {

    const auto code = PointToCode(h_data2[i]);

    std::cout << i << ":\t" << h_data2[i] << "\t" << output_data[i];
    std::cout << '\t' << code;
    std::cout << '\n';
  }

  std::cout << "Done\n";
  return EXIT_SUCCESS;
}
