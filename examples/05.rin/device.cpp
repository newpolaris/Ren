#include "device.h"
#include <vector>
#include <macro.h>

#if WIN32
#include <windows.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#endif
#include <glfw/glfw3.h>
#include <glfw/glfw3native.h>

VkDevice CreateDevice(VkPhysicalDevice physical_device, uint32_t queue_family_index) {
    std::vector<const char*> device_extensions {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
        VK_KHR_16BIT_STORAGE_EXTENSION_NAME,
        VK_KHR_8BIT_STORAGE_EXTENSION_NAME,
        VK_KHR_DRAW_INDIRECT_COUNT_EXTENSION_NAME,
    };

    const float qeue_priorites[] = { 1.f };
    VkDeviceQueueCreateInfo queue_create_info = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO  };
    queue_create_info.queueFamilyIndex = queue_family_index;
    queue_create_info.queueCount = ARRAY_SIZE(qeue_priorites);
    queue_create_info.pQueuePriorities = qeue_priorites;

    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extensionCount, nullptr);

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extensionCount, availableExtensions.data());

    VkPhysicalDeviceFeatures2 features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
    features.features.multiDrawIndirect = true;

    VkPhysicalDevice8BitStorageFeaturesKHR features8 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES_KHR };
    features8.storageBuffer8BitAccess = true;
    features8.uniformAndStorageBuffer8BitAccess = true;

    VkPhysicalDevice16BitStorageFeatures features16 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES_KHR };
    features16.storageBuffer16BitAccess = true;
    features16.uniformAndStorageBuffer16BitAccess = true;

    VkPhysicalDeviceMeshShaderFeaturesNV featuresMesh = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_NV };
    featuresMesh.meshShader = true;

    features.pNext = &features16;
    features16.pNext = &features8;

    VkDeviceCreateInfo info = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    info.queueCreateInfoCount = 1;
    info.pQueueCreateInfos = &queue_create_info;
    info.enabledExtensionCount = static_cast<uint32_t>(device_extensions.size());;
    info.ppEnabledExtensionNames = device_extensions.data();
    info.pNext = &features;

    VkDevice device = VK_NULL_HANDLE;
    VK_ASSERT(vkCreateDevice(physical_device, &info, nullptr, &device));
    return device;
}

VkSurfaceKHR CreateSurface(VkInstance instance, void* pointer) {
    VkSurfaceKHR surface = VK_NULL_HANDLE;

#ifdef WIN32
    GLFWwindow* windows = reinterpret_cast<GLFWwindow*>(pointer);
    ASSERT(windows);
    VkWin32SurfaceCreateInfoKHR info = { VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR };
    info.hinstance = GetModuleHandle(NULL);
    info.hwnd = glfwGetWin32Window(windows);

    VK_ASSERT(vkCreateWin32SurfaceKHR(instance, &info, nullptr, &surface));
#else
    #error
#endif

    return surface;
}

VkPhysicalDevice CreatePhysicalDevice(VkInstance instance) {
    uint32_t device_count = 0;
    VK_ASSERT(vkEnumeratePhysicalDevices(instance, &device_count, nullptr));
    std::vector<VkPhysicalDevice> devices(device_count);
    VK_ASSERT(vkEnumeratePhysicalDevices(instance, &device_count, devices.data()));
    
    for (auto device : devices)
    {
        VkPhysicalDeviceProperties properties = {};
        vkGetPhysicalDeviceProperties(device, &properties);
        if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
            return device;
    }
    return devices.front();
}

PhysicalDeviceProperties CreatePhysicalDeviceProperties(VkPhysicalDevice device)
{
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(device, &properties);

    VkPhysicalDeviceMemoryProperties memory;
    vkGetPhysicalDeviceMemoryProperties(device, &memory);

    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
    QueueFamilyProperties queue(count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, queue.data());

    return PhysicalDeviceProperties { properties, memory, queue };
}

// Choose graphics queue that also support present
uint32_t GetQueueFamilyIndex(VkPhysicalDevice device, const QueueFamilyProperties& properties, VkSurfaceKHR surface) {
    const auto& queue_properties = properties;

    for (size_t i = 0; i < properties.size(); i++) {
        VkBool32 support = VK_FALSE;
        VK_ASSERT(vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &support));

        if ((properties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && properties[i].queueCount > 0)
            return i;
    }
    return VK_QUEUE_FAMILY_IGNORED;
}

