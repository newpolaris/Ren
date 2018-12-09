#pragma once

#include <volk.h>

struct BufferCreateinfo {
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

Buffer CreateBuffer(VkDevice device, const VkPhysicalDeviceMemoryProperties& properties, const BufferCreateinfo& info);
void DestroyBuffer(VkDevice device, Buffer* buffer);
void UploadBuffer(VkDevice device, VkCommandPool pool, VkQueue queue, const Buffer& staging, const Buffer& dst, 
                  VkDeviceSize size, const void* data);
