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

VkInstance CreateInstance(const char* ApplicationName, const char* EngineName) {

    const char* validation_layers[] = {
        "VK_LAYER_LUNARG_standard_validation",
    };
     
    const char* extension_names[] = {
        VK_KHR_SURFACE_EXTENSION_NAME,
    #ifdef WIN32
        VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
    #else
        #error
    #endif
    #if _DEBUG
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
    #endif
    };

    VkApplicationInfo app_info = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app_info.pApplicationName = ApplicationName;
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = EngineName;
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo info = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    info.pApplicationInfo = &app_info;
    #ifdef _DEBUG
    info.enabledLayerCount = ARRAY_SIZE(validation_layers);
    info.ppEnabledLayerNames = validation_layers;
    #endif
    info.enabledExtensionCount = ARRAY_SIZE(extension_names);
    info.ppEnabledExtensionNames = extension_names;

    VkInstance instance = VK_NULL_HANDLE;
    VK_ASSERT(vkCreateInstance(&info, nullptr, &instance));
    return instance;
}

VkDebugUtilsMessengerEXT CreateDebugCallback(VkInstance instance, PFN_vkDebugUtilsMessengerCallbackEXT debugcallback) {
    if (vkCreateDebugUtilsMessengerEXT == nullptr)
        return VK_NULL_HANDLE;

    VkDebugUtilsMessengerCreateInfoEXT info = {VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    info.pfnUserCallback = debugcallback;

    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    VK_ASSERT(vkCreateDebugUtilsMessengerEXT(instance, &info, nullptr, &messenger));
    return messenger;
}

void DestroyDebugCallback(VkInstance instance, VkDebugUtilsMessengerEXT messenger) {
    if (vkDestroyDebugUtilsMessengerEXT == nullptr)
        return;
    vkDestroyDebugUtilsMessengerEXT(instance, messenger, nullptr);
}

VkSurfaceFormatKHR GetSurfaceFormat(VkPhysicalDevice physical_device, VkSurfaceKHR surface) {
    uint32_t formatcount = 0;
    VK_ASSERT(vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &formatcount, nullptr));
    std::vector<VkSurfaceFormatKHR> surfaceformats(formatcount);
    VK_ASSERT(vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &formatcount, surfaceformats.data()));

    // Test if available
    VkSurfaceFormatKHR surfaceformat = { VK_FORMAT_UNDEFINED, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
    for (auto format : surfaceformats)
    {
        if (format.format == VK_FORMAT_B8G8R8A8_UNORM && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            surfaceformat = format;
            break;
        }
    }
    ASSERT(surfaceformat.format != VK_FORMAT_UNDEFINED);
    return surfaceformat;
}

SurfaceProperties CreateSurfaceProperties(VkPhysicalDevice physical_device, const QueueFamilyProperties& properties,
                                          VkSurfaceKHR surface) {
    VkSurfaceFormatKHR swapchain_format = GetSurfaceFormat(physical_device, surface);

    uint32_t presentcount = 0;
    VK_ASSERT(vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface, &presentcount, nullptr));
    std::vector<VkPresentModeKHR> presentmodes(presentcount);
    VK_ASSERT(vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface, &presentcount, presentmodes.data()));

    VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;
    for (auto mode : presentmodes) {
        if (mode == VK_PRESENT_MODE_MAILBOX_KHR) {
            present_mode = mode;
            break;
        }
    }

    const uint32_t queue_family_index = GetQueueFamilyIndex(physical_device, properties, surface);
    ASSERT(queue_family_index != VK_QUEUE_FAMILY_IGNORED);

    return { swapchain_format, present_mode, queue_family_index };
}

