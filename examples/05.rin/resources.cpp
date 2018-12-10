#include "resources.h"

#include <macro.h>
#include <cstdio>
#include <bits.h>

#include "synchronizes.h"

Buffer CreateBuffer(VkDevice device, const VkPhysicalDeviceMemoryProperties& properties, const BufferCreateInfo& info) {
    ASSERT(info.size);

    VkBufferCreateInfo create_info = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    create_info.size = info.size;
    create_info.usage = info.usage;
    create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer = VK_NULL_HANDLE;
    VK_ASSERT(vkCreateBuffer(device, &create_info, nullptr, &buffer));

    VkMemoryRequirements requirement;
    vkGetBufferMemoryRequirements(device, buffer, &requirement);

    uint32_t type_index = 0;
    uint32_t heap_index = 0;
    for (uint32_t i = 0; i < properties.memoryTypeCount; i++) {
        if (flag_test(properties.memoryTypes[i].propertyFlags, info.flags) &&
            bit_test(requirement.memoryTypeBits, i)) {
            type_index = i;
            heap_index = properties.memoryTypes[i].heapIndex; 
        }
    }

    VkMemoryAllocateInfo alloc = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    alloc.memoryTypeIndex = type_index;
    alloc.allocationSize = requirement.size;

    VkDeviceMemory memory = VK_NULL_HANDLE;
    VK_ASSERT(vkAllocateMemory(device, &alloc, nullptr, &memory));

    VkDeviceSize offset = 0;
    VK_ASSERT(vkBindBufferMemory(device, buffer, memory, offset));

    void* data = nullptr;
    if (info.flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        VK_ASSERT(vkMapMemory(device, memory, 0, requirement.size, 0, &data));
        if (info.data)
            memcpy(data, info.data, static_cast<uint32_t>(info.size));
    }

    return Buffer { buffer, memory, info.flags, info.usage, requirement.size, data };
}

void DestroyBuffer(VkDevice device, Buffer* buffer) {
    ASSERT(buffer);

    vkFreeMemory(device, buffer->memory, nullptr);
    buffer->memory = VK_NULL_HANDLE;
    vkDestroyBuffer(device, buffer->buffer, nullptr);
    buffer->buffer = VK_NULL_HANDLE;
    buffer->data = nullptr;
}

void CopyBuffer(VkDevice device, VkCommandPool pool, VkQueue queue, 
                const Buffer& src, VkDeviceSize size, const Buffer& dst)
{
    VkCommandBufferAllocateInfo alloc = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandPool = pool;
    alloc.commandBufferCount = 1;

    VkCommandBuffer command = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(device, &alloc, &command);

    VK_ASSERT(vkResetCommandBuffer(command, 0));
    VkCommandBufferBeginInfo begin { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_ASSERT(vkBeginCommandBuffer(command, &begin));
    
    VkBufferCopy resion = { 0, 0, size };
    vkCmdCopyBuffer(command, src.buffer, dst.buffer, 1, &resion);
    VK_ASSERT(vkEndCommandBuffer(command));

    VkSubmitInfo submit = {};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;

    VkFence fence = CreateFence(device, 0);
    VK_ASSERT(vkQueueSubmit(queue, 1, &submit, fence));
    VK_ASSERT(vkWaitForFences(device, 1, &fence, TRUE, ~0ull));
    vkDestroyFence(device, fence, nullptr);
}

void UploadBuffer(VkDevice device, VkCommandPool pool, VkQueue queue,
                  const Buffer& staging, const Buffer& dst, VkDeviceSize size, const void* data)
{
    ASSERT(size <= dst.size);
    ASSERT(staging.data && data);

    memcpy(staging.data, data, static_cast<size_t>(size));
    
    CopyBuffer(device, pool, queue, staging, size, dst); 
}

Image CreateImage(VkDevice device, const VkPhysicalDeviceMemoryProperties& properties, const ImageCreateInfo& info) {
    VkImageCreateInfo create_info = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    create_info.imageType = info.type;
    create_info.format = info.format;
    create_info.extent = { info.extent.width, info.extent.height, 1 };
    create_info.mipLevels = info.mips;
    create_info.arrayLayers = info.arrays;
    create_info.samples = VK_SAMPLE_COUNT_1_BIT;
    create_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    create_info.usage = info.usage;
    create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    create_info.initialLayout = info.layout;

    VkImage image = VK_NULL_HANDLE;
    VK_ASSERT(vkCreateImage(device, &create_info, nullptr, &image));

    VkMemoryRequirements requirement;
    vkGetImageMemoryRequirements(device, image, &requirement);

    uint32_t type_index = 0;
    uint32_t heap_index = 0;
    for (uint32_t i = 0; i < properties.memoryTypeCount; i++) {
        if (flag_test(properties.memoryTypes[i].propertyFlags, info.flags) &&
            bit_test(requirement.memoryTypeBits, i)) {
            type_index = i;
            heap_index = properties.memoryTypes[i].heapIndex; 
        }
    }

    VkMemoryAllocateInfo alloc = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    alloc.memoryTypeIndex = type_index;
    alloc.allocationSize = requirement.size;

    VkDeviceMemory memory = VK_NULL_HANDLE;
    VK_ASSERT(vkAllocateMemory(device, &alloc, nullptr, &memory));

    VkDeviceSize offset = 0;
    VK_ASSERT(vkBindImageMemory(device, image, memory, offset));

    VkImageView view = CreateImageView(device, image, info.format);
    return Image { image, memory, view, info };
}

VkImageView CreateImageView(VkDevice device, VkImage image, VkFormat format) {
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    if (IsDepthFormat(format))
        aspect = VK_IMAGE_ASPECT_DEPTH_BIT;

    VkComponentMapping components {
        VK_COMPONENT_SWIZZLE_IDENTITY,
        VK_COMPONENT_SWIZZLE_IDENTITY,
        VK_COMPONENT_SWIZZLE_IDENTITY,
        VK_COMPONENT_SWIZZLE_IDENTITY
    };

    VkImageSubresourceRange range {
        aspect,
        0, 1, 0, 1
    };

    VkImageViewCreateInfo info = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    info.image = image;
    info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    info.format = format;
    info.components = components;
    info.subresourceRange = range;

    VkImageView view = VK_NULL_HANDLE;
    VK_ASSERT(vkCreateImageView(device, &info, nullptr, &view));
    return view;
}

void DestroyImage(VkDevice device, Image* image)
{
    vkDestroyImageView(device, image->view, nullptr);
    image->view = VK_NULL_HANDLE;
    vkDestroyImage(device, image->image, nullptr);
    image->image = VK_NULL_HANDLE;
    vkFreeMemory(device, image->memory, nullptr);
    image->memory = VK_NULL_HANDLE;
}

VkImageMemoryBarrier CreateImageBarrier(VkImage image, VkAccessFlags src, VkAccessFlags dst,
                                        VkImageLayout old_layout, VkImageLayout new_layout,
                                        VkImageAspectFlags aspect ) {
    VkImageMemoryBarrier barrier { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    barrier.srcAccessMask = src;
    barrier.dstAccessMask = dst;
    barrier.oldLayout = old_layout;
    barrier.newLayout = new_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = aspect;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS ;

    return barrier;
}
