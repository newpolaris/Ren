#pragma once

#include <volk.h>

struct BufferCreateInfo {
    VkBufferUsageFlags usage;
    VkMemoryPropertyFlags flags;
    VkDeviceSize size;
    const void* data;
};

struct Buffer {
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkBufferUsageFlags usage;
    VkMemoryPropertyFlags flags;
    VkDeviceSize size;
    void* data;
};

Buffer CreateBuffer(VkDevice device, const VkPhysicalDeviceMemoryProperties& properties, const BufferCreateInfo& info);
void DestroyBuffer(VkDevice device, Buffer* buffer);
void UploadBuffer(VkDevice device, VkCommandPool pool, VkQueue queue, const Buffer& staging, const Buffer& dst, 
                  VkDeviceSize size, const void* data);

inline bool IsDepthFormat(VkFormat format)
{
    switch (format) {
    case VK_FORMAT_D16_UNORM:
    case VK_FORMAT_X8_D24_UNORM_PACK32:
    case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_D16_UNORM_S8_UINT:
    case VK_FORMAT_D24_UNORM_S8_UINT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
        return true;
    }
    return false;
}

struct ImageCreateInfo {
    VkImageUsageFlags usage;
    VkMemoryPropertyFlags flags;
    VkFormat format;
    VkExtent2D extent;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageType type = VK_IMAGE_TYPE_2D;
    uint32_t mips = 1;
    uint32_t arrays = 1;
};

struct Image {
    VkImage image;
    VkDeviceMemory memory;
    VkImageView view;
    ImageCreateInfo info;
};
 
Image CreateImage(VkDevice device, const VkPhysicalDeviceMemoryProperties& properties, const ImageCreateInfo& info);
void DestroyImage(VkDevice device, Image* image);
VkImageView CreateImageView(VkDevice device, VkImage image, VkFormat format);
VkImageMemoryBarrier CreateImageBarrier(VkImage image, VkAccessFlags src, VkAccessFlags dst,
                                        VkImageLayout old_layout, VkImageLayout new_layout, VkImageAspectFlags aspect);
