#include "resources.h"

#include <macro.h>
#include <cstdio>
#include <bits.h>

#include "synchronizes.h"

Buffer CreateBuffer(VkDevice device, const VkPhysicalDeviceMemoryProperties& properties, const BufferCreateinfo& info) {
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

