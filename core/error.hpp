#pragma once

#include <stdexcept>
#include <vulkan/vulkan_core.h>

class VulkanException : public std::runtime_error {
public:
  /**
   * @brief Vulkan exception constructor
   */
  VulkanException(VkResult result, const std::string &msg = "Vulkan error");

  /**
   * @brief Returns the Vulkan error code as string
   * @return String message of exception
   */
  const char *what() const noexcept override;

  VkResult result;

private:
  std::string error_message;
};
