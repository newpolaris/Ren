#include "synchronizes.h"

#include <macro.h>

VkSemaphore CreateSemaphore(VkDevice device, VkSemaphoreCreateFlags flags) {
    VkSemaphoreCreateInfo info = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    info.flags = flags;
    VkSemaphore semaphore = VK_NULL_HANDLE;
    VK_ASSERT(vkCreateSemaphore(device, &info, nullptr, &semaphore));
    return semaphore;
}

std::vector<VkSemaphore> CreateSemaphore(VkDevice device, VkSemaphoreCreateFlags flags, size_t nums) {
    std::vector<VkSemaphore> semaphores;
    for (size_t i = 0; i < nums; i++) {
        VkSemaphoreCreateInfo info = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        info.flags = flags;
        VkSemaphore semaphore = VK_NULL_HANDLE;
        VK_ASSERT(vkCreateSemaphore(device, &info, nullptr, &semaphore));
        semaphores.push_back(semaphore);
    }
    return semaphores;
}

VkFence CreateFence(VkDevice device, VkFenceCreateFlags flags) {
    VkFenceCreateInfo info { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    info.flags = flags;
    VkFence fence = VK_NULL_HANDLE;
    VK_ASSERT(vkCreateFence(device, &info, nullptr, &fence));
    return fence;
}

std::vector<VkFence> CreateFence(VkDevice device, VkFenceCreateFlags flags, size_t nums) {
    std::vector<VkFence> fences;
    for (size_t i = 0; i < nums; i++) {
        VkFenceCreateInfo info { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        info.flags = flags;
        VkFence fence = VK_NULL_HANDLE;
        VK_ASSERT(vkCreateFence(device, &info, nullptr, &fence));
        fences.push_back(fence);
    }
    return fences;
}
