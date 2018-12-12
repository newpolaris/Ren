#pragma once

#include "types.h"
#include "device.h"

struct Swapchain {
    VkExtent2D extent;
    VkSwapchainKHR swapchain;
    std::vector<VkImage> images;
};

void CreateSwapchain(VkDevice device, const VkSurfaceCapabilitiesKHR& capabilities, VkSurfaceKHR surface,
                     const SurfaceProperties& properties, VkRenderPass renderpass, Swapchain* result);
void DestroySwapchain(VkDevice device, Swapchain* result);

