#pragma once

#include <volk.h>
#include <vector>

#if WIN32
#undef CreateSemaphore
#endif

VkSemaphore CreateSemaphore(VkDevice device, VkSemaphoreCreateFlags flags);
VkFence CreateFence(VkDevice device, VkFenceCreateFlags flags);

std::vector<VkSemaphore> CreateSemaphore(VkDevice device, VkSemaphoreCreateFlags flags, size_t nums);
std::vector<VkFence> CreateFence(VkDevice device, VkFenceCreateFlags flags, size_t nums);
