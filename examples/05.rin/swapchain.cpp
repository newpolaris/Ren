#include "swapchain.h"
#include <macro.h>
#include <algorithm>

VkSwapchainKHR CreateSwapchain(VkDevice device, VkSurfaceKHR surface, const SurfaceProperties& properties,
                               const VkSurfaceCapabilitiesKHR& capabilities, VkSwapchainKHR oldswapchain) {
    constexpr VkCompositeAlphaFlagBitsKHR composite_alpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    constexpr VkSurfaceTransformFlagBitsKHR pretransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;

    ASSERT(capabilities.supportedTransforms & pretransform);
    ASSERT(capabilities.supportedCompositeAlpha & composite_alpha);

    const uint32_t imagecount = std::min(capabilities.minImageCount + 1, capabilities.maxImageCount);

    // The Vulkan spec states: imageExtent must be between minImageExtent and maxImageExtent,
    VkExtent2D currentExtent = capabilities.currentExtent;

    VkSwapchainCreateInfoKHR info = { VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
    info.surface = surface;
    info.minImageCount = imagecount;
    info.imageFormat = properties.format.format;
    info.imageColorSpace = properties.format.colorSpace;
    info.imageExtent = currentExtent;
    info.imageArrayLayers = std::min(1u, capabilities.maxImageArrayLayers);
    info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    info.compositeAlpha = composite_alpha;
    info.presentMode = properties.present_mode;
    info.clipped = VK_TRUE;
    info.oldSwapchain = oldswapchain;

    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VK_ASSERT(vkCreateSwapchainKHR(device, &info, nullptr, &swapchain));
    return swapchain;
}

void CreateSwapchain(VkDevice device, const VkSurfaceCapabilitiesKHR& capabilities, VkSurfaceKHR surface,
                     const SurfaceProperties& properties, VkRenderPass renderpass, Swapchain* result) {
    ASSERT(result);

    // recreate new swapchain with old swapchain
    VkSwapchainKHR swapchain = CreateSwapchain(device, surface, properties, capabilities, result->swapchain);

    // destroy old swapchain
    DestroySwapchain(device, result);

    VkExtent2D extent = capabilities.currentExtent;

    uint32_t count = 0;
    VK_ASSERT(vkGetSwapchainImagesKHR(device, swapchain, &count, nullptr));
    std::vector<VkImage> images(count);
    VK_ASSERT(vkGetSwapchainImagesKHR(device, swapchain, &count, images.data()));

    *result = Swapchain { extent, swapchain, images };
}

void DestroySwapchain(VkDevice device, Swapchain* result) {
    ASSERT(result);

    vkDestroySwapchainKHR(device, result->swapchain, nullptr);
    result->swapchain = VK_NULL_HANDLE;
}
