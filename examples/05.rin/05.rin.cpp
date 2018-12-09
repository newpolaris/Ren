// build vulkan renderer from scratch

#if WIN32
#include <windows.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#endif

#include <volk/volk.h>
#include <glfw/glfw3.h>
#include <glfw/glfw3native.h>
#include <vulkan/vulkan.h>
#include <cassert>
#include <cstdlib>
#include <vector>
#include <set>
#include <optional>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <glm/glm.hpp>

#include "mesh.h"
#include "macro.h"
#include "shader_module.h"
#include "program.h"
#include "synchronizes.h"
#include "resources.h"

#define SUPPORT_MULTIFRAME_IN_FLIGHT 0
#define CLUSTER_CULL 1

VkInstance CreateInstance(
    const char* ApplicationName,
    const char* EngineName) {

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
    info.enabledLayerCount = ARRAY_SIZE(validation_layers);
    info.ppEnabledLayerNames = validation_layers;
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

VkSurfaceKHR CreateSurface(VkInstance instance, GLFWwindow* windows) {
    VkSurfaceKHR surface = VK_NULL_HANDLE;

#ifdef WIN32
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

using QueueFamilyProperties = std::vector<VkQueueFamilyProperties>;

struct PhysicalDeviceProperties {
    VkPhysicalDeviceProperties properties;
    VkPhysicalDeviceMemoryProperties memory;
    QueueFamilyProperties queue;
};

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

VkDevice CreateDevice(VkPhysicalDevice physical_device, uint32_t queue_family_index) {
    std::vector<const char*> device_extensions {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
        VK_KHR_16BIT_STORAGE_EXTENSION_NAME,
        VK_KHR_8BIT_STORAGE_EXTENSION_NAME,
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

VkRenderPass CreateRenderPass(VkDevice device, VkFormat format) {
    VkAttachmentDescription description = {};
    description.format = format;
    description.samples = VK_SAMPLE_COUNT_1_BIT;
    description.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    description.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    description.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    description.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    description.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED ;
    description.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference color_attachment_ref = {};
    color_attachment_ref.attachment = 0;
    color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_attachment_ref;

    VkRenderPassCreateInfo info = { VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO };
    info.attachmentCount = 1;
    info.pAttachments = &description;
    info.subpassCount = 1;
    info.pSubpasses = &subpass;

    VkRenderPass renderpass = VK_NULL_HANDLE;
    VK_ASSERT(vkCreateRenderPass(device, &info, nullptr, &renderpass));
    return renderpass;
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

struct SurfaceProperties {
    VkSurfaceFormatKHR format;
    VkPresentModeKHR present_mode;
    uint32_t queue_family_index;
};

SurfaceProperties CreateSurfaceProperties(VkPhysicalDevice physical_device, const QueueFamilyProperties& properties, VkSurfaceKHR surface) {
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

VkSwapchainKHR CreateSwapchain(VkDevice device, VkSurfaceKHR surface, const SurfaceProperties& properties,
                               const VkSurfaceCapabilitiesKHR& capabilities, VkSwapchainKHR oldswapchain) {
    constexpr VkCompositeAlphaFlagBitsKHR composite_alpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    constexpr VkSurfaceTransformFlagBitsKHR pretransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;

    ASSERT(capabilities.supportedTransforms & pretransform);
    ASSERT(capabilities.supportedCompositeAlpha & composite_alpha);

    uint32_t imagecount = std::min(capabilities.minImageCount + 1, capabilities.maxImageCount);

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
    info.queueFamilyIndexCount = properties.queue_family_index;
    info.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    info.compositeAlpha = composite_alpha;
    info.presentMode = properties.present_mode;
    info.clipped = VK_TRUE;
    info.oldSwapchain = oldswapchain;

    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VK_ASSERT(vkCreateSwapchainKHR(device, &info, nullptr, &swapchain));
    return swapchain;
}

VkFramebuffer CreateFramebuffer(VkDevice device, VkRenderPass pass, VkImageView view, VkExtent2D extent) {
    VkFramebufferCreateInfo info = { VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
    info.renderPass = pass;
    info.attachmentCount = 1;
    info.pAttachments = &view;
    info.width = extent.width;
    info.height = extent.height;
    info.layers = 1;

    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    VK_ASSERT(vkCreateFramebuffer(device, &info, nullptr, &framebuffer));
    return framebuffer;
}

VkImageView CreateImageView(VkDevice device, VkImage image, VkFormat format) {
    VkComponentMapping components {
        VK_COMPONENT_SWIZZLE_IDENTITY,
        VK_COMPONENT_SWIZZLE_IDENTITY,
        VK_COMPONENT_SWIZZLE_IDENTITY,
        VK_COMPONENT_SWIZZLE_IDENTITY
    };

    VkImageSubresourceRange range {
        VK_IMAGE_ASPECT_COLOR_BIT,
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

struct Swapchain {
    VkExtent2D extent;
    VkSwapchainKHR swapchain;
    std::vector<VkImageView> imageviews;
    std::vector<VkFramebuffer> framebuffers;
};

void DestroySwapchain(VkDevice device, Swapchain* result) {
    ASSERT(result);

    for (auto view : result->imageviews)
        vkDestroyImageView(device, view, nullptr);
    result->imageviews.clear();

    for (auto framebuffer : result->framebuffers)
        vkDestroyFramebuffer(device, framebuffer, nullptr);
    result->framebuffers.clear();

    vkDestroySwapchainKHR(device, result->swapchain, nullptr);
    result->swapchain = VK_NULL_HANDLE;
}

void CreateSwapchain(VkDevice device, const VkSurfaceCapabilitiesKHR& capabilities, VkSurfaceKHR surface,
                     const SurfaceProperties& properties, VkRenderPass renderpass, Swapchain* result) {
    ASSERT(result);

    VkSwapchainKHR swapchain = CreateSwapchain(device, surface, properties, capabilities, result->swapchain);

    DestroySwapchain(device, result);

    VkExtent2D extent = capabilities.currentExtent;

    uint32_t count = 0;
    VK_ASSERT(vkGetSwapchainImagesKHR(device, swapchain, &count, nullptr));
    std::vector<VkImage> images(count);
    VK_ASSERT(vkGetSwapchainImagesKHR(device, swapchain, &count, images.data()));

    std::vector<VkImageView> imageviews;
    for (auto image : images) 
        imageviews.push_back(CreateImageView(device, image, properties.format.format));

    std::vector<VkFramebuffer> framebuffers;
    for (auto view : imageviews)
        framebuffers.push_back(CreateFramebuffer(device, renderpass, view, extent));

    *result = Swapchain { extent, swapchain, imageviews, framebuffers };
}

void UpdateViewportScissor(VkExtent2D extent, VkViewport* viewport, VkRect2D* scissor) {
    // viewport trick to make positive y upward 
    viewport->x = 0.f;
    viewport->y = static_cast<float>(extent.height);
    viewport->width = static_cast<float>(extent.width);
    viewport->height = -static_cast<float>(extent.height);
    viewport->minDepth = 0.f;
    viewport->maxDepth = 1.f;

    scissor->offset = VkOffset2D { 0, 0 };
    scissor->extent = extent;
}

// UNDONE; TODO;
VkDescriptorPool CreateDescriptorPool(VkDevice device) {
    VkDescriptorPoolCreateInfo info { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VK_ASSERT(vkCreateDescriptorPool(device, &info, nullptr, &pool));
    return pool;
}

// UNDONE; TODO;
VkDescriptorSet CreateDescriptorSet(VkDevice device, VkDescriptorPool pool) {
    VkDescriptorSetAllocateInfo info = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    info.descriptorPool = pool;
    info.descriptorSetCount = 1;

    VkDescriptorSet sets = VK_NULL_HANDLE;
    VK_ASSERT(vkAllocateDescriptorSets(device, &info, &sets));
    return sets;
}

VkCommandBuffer CreateCommandBuffer(VkDevice device, VkCommandPool pool) {
    VkCommandBufferAllocateInfo info { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    info.commandPool = pool;
    info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    info.commandBufferCount = 1;

    VkCommandBuffer buffer = VK_NULL_HANDLE;
    VK_ASSERT(vkAllocateCommandBuffers(device, &info, &buffer));
    return buffer;
}

std::vector<VkCommandBuffer> CreateCommandBuffer(VkDevice device, VkCommandPool pool, size_t num_chains) {
    VkCommandBufferAllocateInfo info { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    info.commandPool = pool;
    info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    info.commandBufferCount = num_chains;

    std::vector<VkCommandBuffer> buffer(num_chains);
    VK_ASSERT(vkAllocateCommandBuffers(device, &info, buffer.data()));
    return buffer;
}

VkQueryPool CreateQueryPool(VkDevice device, VkQueryType type, uint32_t count, 
                            VkQueryPipelineStatisticFlags statistics) {
    VkQueryPoolCreateInfo info = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    info.queryType = type;
    info.queryCount = count;
    info.pipelineStatistics = statistics;
    VkQueryPool pool = VK_NULL_HANDLE;
    VK_ASSERT(vkCreateQueryPool(device, &info, nullptr, &pool));
    return pool;
}

// Create on indirect command for each mesh in the scene
std::vector<VkDrawIndexedIndirectCommand> CreateIndirectCommandBuffer(const Mesh& mesh) {
    std::vector<VkDrawIndexedIndirectCommand> commands;

    for (size_t i = 0; i < mesh.meshlet_instances.size(); i++)
    {
        auto cone = mesh.meshlets[i].cone;
        auto cosangle = glm::dot(glm::vec3(cone[0], cone[1], cone[2]), glm::vec3(0, 0, 1));
        if (cone[3] < cosangle)
            continue;

        VkDrawIndexedIndirectCommand indirectCmd {};
        indirectCmd.instanceCount = 1;
        indirectCmd.firstInstance = uint32_t(i);
        indirectCmd.firstIndex = mesh.meshlet_instances[i].first;
        indirectCmd.indexCount = mesh.meshlet_instances[i].second;
        
        commands.push_back(indirectCmd);
    }
    return std::move(commands);
}

static VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                    VkDebugUtilsMessageTypeFlagsEXT messageType,
                                                    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                                    void* pUserData) {
    // Validation layers don't correctly track set assignments when using push descriptors with update templates:
    // https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/341
    if (strstr(pCallbackData->pMessage, "uses set #0 but that set is not bound."))
        return VK_FALSE;

    // Validation layers don't correctly detect NonWriteable declarations for storage buffers:
    // https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/73
    if (strstr(pCallbackData->pMessage, 
               "Shader requires vertexPipelineStoresAndAtomics but is not enabled on the device"))
        return VK_FALSE;

    const char* type =
        (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
        ? "ERROR"
        : (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        ? "WARNING"
        : "INFO";

    std::stringstream message;
    message << type << ": " << pCallbackData->pMessage << "\n";

#ifdef _WIN32
    OutputDebugStringA(message.str().c_str());
#endif
    std::cerr << message.str();

    if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
        ASSERT(!"Validation error occurs!");
    return VK_FALSE;
}

int main() {
    const char* application_name = "Hello Kitty";
    const char* engine_name = "Rin";
    const int width = 1280;
    const int height = 1024;

    if (glfwInit() != GLFW_TRUE)
        return EXIT_FAILURE;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    volkInitialize();

    VkInstance instance = CreateInstance(application_name, engine_name);
    if (instance == VK_NULL_HANDLE)
        return EXIT_FAILURE;

    volkLoadInstance(instance);

    VkDebugUtilsMessengerEXT messenger = CreateDebugCallback(instance, DebugCallback);

    GLFWwindow* windows = glfwCreateWindow(width, height, application_name, nullptr, nullptr);

    VkSurfaceKHR surface = CreateSurface(instance, windows);

    VkPhysicalDevice physical_device = CreatePhysicalDevice(instance);

    const auto physical_properties = CreatePhysicalDeviceProperties(physical_device);
    const auto surface_properties = CreateSurfaceProperties(physical_device, physical_properties.queue, surface);

    VkDevice device = CreateDevice(physical_device, surface_properties.queue_family_index);

    VkSurfaceCapabilitiesKHR capabilities = {};
    VK_ASSERT(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &capabilities));
    const VkRenderPass renderpass = CreateRenderPass(device, surface_properties.format.format);

    Swapchain swapchain = {};
    CreateSwapchain(device, capabilities, surface, surface_properties, renderpass, &swapchain);
    
    VkViewport viewport = {};
    VkRect2D scissor = {};

    UpdateViewportScissor(swapchain.extent, &viewport, &scissor); 

    constexpr uint32_t queue_index = 0;
    VkQueue command_queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, surface_properties.queue_family_index, queue_index, &command_queue);
    VkQueue present_queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, surface_properties.queue_family_index, queue_index, &present_queue);

    ShaderModule vertex_shader = CreateShaderModule(device, "shaders/05.rin/base.vert.spv");
    ShaderModule fragment_shader = CreateShaderModule(device, "shaders/05.rin/base.frag.spv");
    ShaderModules shaders = { vertex_shader, fragment_shader };

    VkPipelineLayout layout = CreatePipelineLayout(device, shaders);
    VkPipeline pipeline = CreatePipeline(device, layout, renderpass, shaders);
    VkDescriptorUpdateTemplate descriptorUpdateTemplate = CreateDescriptorUpdateTemplate(device, layout, shaders);

    VkCommandPoolCreateInfo info = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    info.queueFamilyIndex = surface_properties.queue_family_index;

    VkCommandPool command_pool = VK_NULL_HANDLE;
    VK_ASSERT(vkCreateCommandPool(device, &info, nullptr, &command_pool));
    VK_ASSERT(vkResetCommandPool(device, command_pool, 0));

    size_t chains = swapchain.imageviews.size();

    std::vector<VkCommandBuffer> command_buffers = CreateCommandBuffer(device, command_pool, chains);

    enum { kMaxFramesInFight = 2 };

    // cpu-gpu synchronize
    std::vector<VkFence> fences = CreateFence(device, VK_FENCE_CREATE_SIGNALED_BIT, kMaxFramesInFight);

    // gpu-gpu synchronize
    std::vector<VkSemaphore> semaphores = CreateSemaphore(device, 0, kMaxFramesInFight);
    std::vector<VkSemaphore> signal_semaphores = CreateSemaphore(device, 0, kMaxFramesInFight);

#if _DEBUG
    const char* objfile = "models/kitten.obj";
#else
    const char* objfile = "models/buddha.obj";
#endif

    Mesh mesh = LoadMesh(objfile);
    mesh.meshlets = BuildMeshlets(mesh);
    BuildMeshletIndices(&mesh);

    std::vector<VkDrawIndexedIndirectCommand> indirects = CreateIndirectCommandBuffer(mesh);
    uint32_t indirect_draw_count = static_cast<uint32_t>(indirects.size());

    VkMemoryPropertyFlags device_local_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    VkMemoryPropertyFlags host_memory_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
                                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT; 

    const VkDeviceSize vb_size = sizeof(Vertex)*mesh.vertices.size();
    const VkDeviceSize ib_size = sizeof(uint32_t)*mesh.indices.size();
    const VkDeviceSize mib_size = mesh.meshlet_indices.size() * sizeof(uint32_t);
    const VkDeviceSize idcb_size = indirects.size() * sizeof(VkDrawIndexedIndirectCommand);

    Buffer staging = CreateBuffer(device, physical_properties.memory, { 
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, host_memory_flags, 1024*1024*64});

    Buffer vb = CreateBuffer(device, physical_properties.memory, {
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        device_local_flags, vb_size });

    Buffer ib = CreateBuffer(device, physical_properties.memory, {
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        device_local_flags, ib_size });

    Buffer mib = CreateBuffer(device, physical_properties.memory, {
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        device_local_flags, mib_size });

    // muliple indirect draw buffer for drawing culled meshlet cluster
    Buffer idcb = CreateBuffer(device, physical_properties.memory, {
        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        device_local_flags, idcb_size });

    UploadBuffer(device, command_pool, command_queue, staging, vb, vb_size, mesh.vertices.data());
    UploadBuffer(device, command_pool, command_queue, staging, ib, ib_size, mesh.indices.data());
    UploadBuffer(device, command_pool, command_queue, staging, mib, mib_size, mesh.meshlet_indices.data());
    UploadBuffer(device, command_pool, command_queue, staging, idcb, idcb_size, indirects.data());

    VkQueryPool timestamp_pool = CreateQueryPool(device, VK_QUERY_TYPE_TIMESTAMP, 1024, 0);

    uint32_t row_count = 10;
    uint32_t draw_count = row_count*row_count;
    std::vector<MeshDraw> draws(draw_count);
    for (uint32_t i = 0; i < draw_count; i++) {
        draws[i].offset[0] = (float(i % row_count) + 0.5f) / row_count;
        draws[i].offset[1] = (float(i / row_count) + 0.5f) / row_count;
        draws[i].scale[0] = 1.f / row_count;
        draws[i].scale[1] = 1.f / row_count;
    }

    double cpu_average = 0.0, gpu_average = 0.0, wait_average = 0.0;

    size_t current_frame = 0;

    while(!glfwWindowShouldClose(windows)) {
        glfwPollEvents();

        auto fence = fences[current_frame];
    #if SUPPORT_MULTIFRAME_IN_FLIGHT
        VK_ASSERT(vkWaitForFences(device, 1, &fence, TRUE, ~0ull));
    #endif

        auto cpu_begin = std::chrono::steady_clock::now(); 
        auto semaphore = semaphores[current_frame];
        auto signal_semaphore = signal_semaphores[current_frame];

        uint32_t image_index = 0;
        VkResult result = vkAcquireNextImageKHR(device, swapchain.swapchain, ~0ull, semaphore, VK_NULL_HANDLE, &image_index);

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
            VK_ASSERT(vkDeviceWaitIdle(device));
            VK_ASSERT(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &capabilities));
            if (capabilities.currentExtent.height == 0 && capabilities.currentExtent.width == 0)
                continue;
            CreateSwapchain(device, capabilities, surface, surface_properties, renderpass, &swapchain);
            UpdateViewportScissor(swapchain.extent, &viewport, &scissor);
            continue;
        } 
        ASSERT(result == VK_SUCCESS);

        auto command_buffer = command_buffers[image_index];

        VK_ASSERT(vkResetCommandBuffer(command_buffer, 0));
        VkCommandBufferBeginInfo begininfo { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        VK_ASSERT(vkBeginCommandBuffer(command_buffer, &begininfo));

        vkCmdWriteTimestamp(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timestamp_pool, current_frame*2);

        VkClearColorValue clear_color = { std::sin(static_cast<float>(glfwGetTime()))*0.5f + 0.5f, 0.5f, 0.5f, 1.0f };
        const VkClearValue clear_values[] = { clear_color, };

        VkRenderPassBeginInfo pass = { VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
        pass.renderPass = renderpass;
        pass.framebuffer = swapchain.framebuffers[image_index];
        pass.renderArea = scissor;
        pass.clearValueCount = ARRAY_SIZE(clear_values);
        pass.pClearValues = clear_values;

        vkCmdBeginRenderPass(command_buffer, &pass, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        vkCmdSetViewport(command_buffer, 0, 1, &viewport);
        vkCmdSetScissor(command_buffer, 0, 1, &scissor);

        PushDescriptorSets descriptors[] = { vb };
        vkCmdPushDescriptorSetWithTemplateKHR(command_buffer, descriptorUpdateTemplate, layout, 0, &descriptors);

    #if CLUSTER_CULL
        vkCmdBindIndexBuffer(command_buffer, mib.buffer, 0, VK_INDEX_TYPE_UINT32);
    #else
        vkCmdBindIndexBuffer(command_buffer, ib.buffer, 0, VK_INDEX_TYPE_UINT32);
    #endif

        for (auto draw : draws) {
            vkCmdPushConstants(command_buffer, layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(draw), &draw);
        #if CLUSTER_CULL
            vkCmdDrawIndexedIndirect(command_buffer, idcb.buffer, 0, indirect_draw_count, sizeof(VkDrawIndexedIndirectCommand));
        #else
            vkCmdDrawIndexed(command_buffer, static_cast<uint32_t>(mesh.indices.size()), 1, 0, 0, 0);
        #endif
        }
        vkCmdEndRenderPass(command_buffer);

        vkCmdWriteTimestamp(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timestamp_pool, current_frame*2+1);

        VK_ASSERT(vkEndCommandBuffer(command_buffer));

        VkPipelineStageFlags stage_flags[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        VkSubmitInfo submit = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = &semaphore;
        submit.pWaitDstStageMask = stage_flags;
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &signal_semaphore;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command_buffer;

        VK_ASSERT(vkResetFences(device, 1, &fence));
        VK_ASSERT(vkQueueSubmit(command_queue, 1, &submit, fence));

        VkPresentInfoKHR present_info = { VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
        present_info.swapchainCount = 1;
        present_info.pSwapchains = &swapchain.swapchain;
        present_info.pImageIndices = &image_index;
        present_info.pWaitSemaphores = &signal_semaphore;
        present_info.waitSemaphoreCount = 1;
        vkQueuePresentKHR(present_queue, &present_info);

        auto cpu_end = std::chrono::steady_clock::now(); 

    #if SUPPORT_MULTIFRAME_IN_FLIGHT
        auto next_frame = ((current_frame + 1) % kMaxFramesInFight);
    #else
        auto next_frame = current_frame;
    #endif

        // 'VK_QUERY_RESULT_WAIT_BIT' seems not working
        auto wait_begin = std::chrono::steady_clock::now();
        // wait until query result get
        VK_ASSERT(vkWaitForFences(device, 1, &fences[next_frame], TRUE, ~0ull));
        auto wait_end = std::chrono::steady_clock::now();

        uint64_t timestamps[2] = {};
        vkGetQueryPoolResults(device, timestamp_pool, next_frame*2, 2, sizeof(timestamps), timestamps,
                              sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);

        auto wait_time = std::chrono::duration<double, std::milli>(wait_end - wait_begin).count();
        auto cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_begin).count();
        auto gpu_scaler = physical_properties.properties.limits.timestampPeriod;
        auto gpu_time = static_cast<double>((timestamps[1] - timestamps[0]) * 1e-6) * gpu_scaler;

        gpu_average = glm::mix(gpu_average, gpu_time, 0.05);
        cpu_average = glm::mix(cpu_average, cpu_time, 0.05);
        wait_average = glm::mix(wait_average, wait_time, 0.05);

        auto triangle_count = static_cast<int>(mesh.indices.size() / 3);
		auto trianglesPerSec = double(draw_count) * double(triangle_count) / double(gpu_average * 1e-3) * 1e-9;

        char title[256] = {};  
        sprintf(title, "wait: %.2f ms; cpu: %.2f ms; gpu: %.2f ms; triangles %d; meshlets %d; %.1fB tri/sec",
                wait_average, cpu_average, gpu_average, triangle_count, 1, trianglesPerSec);
        glfwSetWindowTitle(windows, title);

    #if SUPPORT_MULTIFRAME_IN_FLIGHT
        current_frame = (current_frame + 1) % kMaxFramesInFight;
    #endif
    }

    VK_ASSERT(vkDeviceWaitIdle(device));
    for (auto fence : fences)
        vkDestroyFence(device, fence, nullptr);
    for (auto semaphore : semaphores)
        vkDestroySemaphore(device, semaphore, nullptr);
    for (auto semaphore : signal_semaphores)
        vkDestroySemaphore(device, semaphore, nullptr);

    DestroyBuffer(device, &staging);
    DestroyBuffer(device, &vb);
    DestroyBuffer(device, &ib);
    DestroyBuffer(device, &mib);
    DestroyBuffer(device, &idcb);

    vkDestroyQueryPool(device, timestamp_pool, nullptr);

    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, layout, nullptr);
    vkDestroyDescriptorUpdateTemplate(device, descriptorUpdateTemplate, nullptr);
    vkDestroyShaderModule(device, vertex_shader.module, nullptr);
    vkDestroyShaderModule(device, fragment_shader.module, nullptr);

    vkDestroyCommandPool(device, command_pool, nullptr);

    DestroySwapchain(device, &swapchain);

    vkDestroyRenderPass(device, renderpass, nullptr);
    vkDestroyDevice(device, nullptr);
    device = VK_NULL_HANDLE;

    DestroyDebugCallback(instance, messenger);

    vkDestroyInstance(instance, nullptr);
    instance = VK_NULL_HANDLE;

    glfwDestroyWindow(windows);
    windows = VK_NULL_HANDLE;

    return 0;
}