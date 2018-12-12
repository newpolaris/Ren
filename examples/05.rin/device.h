#pragma once

#include "types.h"

struct PhysicalDeviceProperties {
    VkPhysicalDeviceProperties properties;
    VkPhysicalDeviceMemoryProperties memory;
    QueueFamilyProperties queue;
};

struct SurfaceProperties {
    VkSurfaceFormatKHR format;
    VkPresentModeKHR present_mode;
    uint32_t queue_family_index;
};

VkDevice CreateDevice(VkPhysicalDevice physical_device, uint32_t queue_family_index);
VkSurfaceKHR CreateSurface(VkInstance instance, void* pointer);
VkPhysicalDevice CreatePhysicalDevice(VkInstance instance);
PhysicalDeviceProperties CreatePhysicalDeviceProperties(VkPhysicalDevice device);
uint32_t GetQueueFamilyIndex(VkPhysicalDevice device, const QueueFamilyProperties& properties, VkSurfaceKHR surface);

