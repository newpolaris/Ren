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

VkInstance CreateInstance(const char* ApplicationName, const char* EngineName);
VkDevice CreateDevice(VkPhysicalDevice physical_device, uint32_t queue_family_index);
VkSurfaceKHR CreateSurface(VkInstance instance, void* pointer);
VkPhysicalDevice CreatePhysicalDevice(VkInstance instance);
PhysicalDeviceProperties CreatePhysicalDeviceProperties(VkPhysicalDevice device);
uint32_t GetQueueFamilyIndex(VkPhysicalDevice device, const QueueFamilyProperties& properties, VkSurfaceKHR surface);
VkDebugUtilsMessengerEXT CreateDebugCallback(VkInstance instance, PFN_vkDebugUtilsMessengerCallbackEXT debugcallback);
void DestroyDebugCallback(VkInstance instance, VkDebugUtilsMessengerEXT messenger);
VkSurfaceFormatKHR GetSurfaceFormat(VkPhysicalDevice physical_device, VkSurfaceKHR surface);
SurfaceProperties CreateSurfaceProperties(VkPhysicalDevice physical_device, const QueueFamilyProperties& properties,
                                          VkSurfaceKHR surface);
