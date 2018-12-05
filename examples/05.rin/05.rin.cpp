// build vulkan renderer from scratch

#if WIN32
#include <windows.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#undef CreateSemaphore
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
#include <objparser.h>
#include <cstdlib>
#include <meshoptimizer.h>
#include <glm/glm.hpp>

#include "macro.h"
#include "shader_module.h"

#define FVF 0

// from boost
template <class integral, class size_t>
constexpr integral align_up(integral x, size_t a) noexcept {
    return integral((x + (integral(a) - 1)) & ~integral(a - 1));
}

template <class integral, class size_t>
constexpr bool bit_test(integral x, size_t bit) noexcept {
    return x & (1 << bit);
}

template <class integral_1, class integral_2>
bool flag_test(integral_1 x, integral_2 flag) noexcept {
    return (x & flag) == flag;
}

struct alignas(16) MeshDraw
{
	float offset[2];
	float scale[2];
};

struct Vertex
{
    uint16_t x, y, z, w;
    uint8_t nx, ny, nz, nw;
    uint16_t tu, tv;
};

struct Mesh
{
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
};

float NegativeIndexHelper(float* src, int i, int sub)
{
    return i < 0 ? 0.f : src[i * 3 + sub];
}

Mesh LoadMesh(const std::string& filename)
{
    ObjFile obj;
    objParseFile(obj, filename.c_str());

    size_t index_count = obj.f_size / 3; 
    std::vector<Vertex> vertices(index_count);

    for (size_t i = 0; i < index_count; i++)
    {
        Vertex& v = vertices[i];

        int vi = obj.f[i * 3 + 0];
        int vti = obj.f[i * 3 + 1];
        int vni = obj.f[i * 3 + 2];

        v.x = meshopt_quantizeHalf(NegativeIndexHelper(obj.v, vi, 0));
        v.y = meshopt_quantizeHalf(NegativeIndexHelper(obj.v, vi, 1));
        v.z = meshopt_quantizeHalf(NegativeIndexHelper(obj.v, vi, 2));
        v.w = 0;
        v.nx = uint8_t(NegativeIndexHelper(obj.vn, vni, 0) * 127.f + 127.f);
        v.ny = uint8_t(NegativeIndexHelper(obj.vn, vni, 1) * 127.f + 127.f);
        v.nz = uint8_t(NegativeIndexHelper(obj.vn, vni, 2) * 127.f + 127.f);
        v.nw = 0;
        v.tu = meshopt_quantizeHalf(NegativeIndexHelper(obj.vt, vti, 0));
        v.tv = meshopt_quantizeHalf(NegativeIndexHelper(obj.vt, vti, 1));
    }

    std::vector<uint32_t> indices(index_count);

    if (false) {
        for (uint32_t i = 0; i < static_cast<uint32_t>(index_count); i++)
            indices[i] = i;
    } else {
        std::vector<uint32_t> remap(index_count);
        size_t vertex_count = meshopt_generateVertexRemap(remap.data(), 0, index_count, vertices.data(), index_count, sizeof(Vertex));

        vertices.resize(vertex_count);

        meshopt_remapVertexBuffer(vertices.data(), vertices.data(), index_count, sizeof(Vertex), remap.data());
        meshopt_remapIndexBuffer(indices.data(), 0, index_count, remap.data());

        meshopt_optimizeVertexCache(indices.data(), indices.data(), index_count, vertex_count);
        meshopt_optimizeVertexFetch(vertices.data(), indices.data(), index_count, vertices.data(), vertex_count, sizeof(Vertex));
    }
    return Mesh { std::move(vertices), std::move(indices) };
}

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

// TODO:
VkDescriptorPool CreateDescriptorPool(VkDevice device) {
    VkDescriptorPoolCreateInfo info { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VK_ASSERT(vkCreateDescriptorPool(device, &info, nullptr, &pool));
    return pool;
}

VkDescriptorSet CreateDescriptorSet(VkDevice device, VkDescriptorPool pool) {
    VkDescriptorSetAllocateInfo info = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    info.descriptorPool = pool;
    info.descriptorSetCount = 1;

    VkDescriptorSet sets = VK_NULL_HANDLE;
    VK_ASSERT(vkAllocateDescriptorSets(device, &info, &sets));
    return sets;
}

VkPipelineLayout CreatePipelineLayout(VkDevice device, ShaderModuleList shaders) {
    std::vector<VkPushConstantRange> ranges = GetPushConstantRange(shaders);

    VkDescriptorSetLayoutBinding set_binding {};
    set_binding.binding = 0;
    set_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    set_binding.descriptorCount = 1;
    set_binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo set_create_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    set_create_info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    set_create_info.bindingCount = 1;
    set_create_info.pBindings = &set_binding;

    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VK_ASSERT(vkCreateDescriptorSetLayout(device, &set_create_info, nullptr, &set_layout));

    VkPipelineLayoutCreateInfo info = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    info.setLayoutCount = 1;
    info.pSetLayouts = &set_layout;
    info.pushConstantRangeCount = static_cast<uint32_t>(ranges.size());
    info.pPushConstantRanges = ranges.data();

    VkPipelineLayout layout = VK_NULL_HANDLE;
    VK_ASSERT(vkCreatePipelineLayout(device, &info, nullptr, &layout));

    vkDestroyDescriptorSetLayout(device, set_layout, nullptr);

    return layout;
}

VkPipeline CreatePipeline(VkDevice device, VkPipelineLayout layout, VkRenderPass pass,
                          const ShaderModule& vs, const ShaderModule& fs) { 
    VkPipelineShaderStageCreateInfo vertex_info = GetPipelineShaderStageCreateInfo(vs, "main");
    VkPipelineShaderStageCreateInfo fragment_info = GetPipelineShaderStageCreateInfo(fs, "main");
    VkPipelineShaderStageCreateInfo shader_stages[] = { vertex_info, fragment_info };

#if FVF
    VkVertexInputBindingDescription bindings[] = { { 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX } };

    VkVertexInputAttributeDescription attributes[] = { 
        { 0, 0, VK_FORMAT_R16G16B16A16_SFLOAT, offsetof(Vertex, x) },
        { 1, 0, VK_FORMAT_R8G8B8A8_UINT, offsetof(Vertex, nx) },
        { 2, 0, VK_FORMAT_R16G16_SFLOAT, offsetof(Vertex, tu) },
    };
#endif

    VkPipelineVertexInputStateCreateInfo vertex_input_info { 
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO 
    };
#if FVF
    vertex_input_info.vertexBindingDescriptionCount = ARRAY_SIZE(bindings);
    vertex_input_info.pVertexBindingDescriptions = bindings;
    vertex_input_info.vertexAttributeDescriptionCount = 3;
    vertex_input_info.pVertexAttributeDescriptions = attributes;
#endif

    VkPipelineInputAssemblyStateCreateInfo input_info { 
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO 
    };
    input_info.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewport_info { VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    viewport_info.viewportCount = 1;
    viewport_info.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo raster_info = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    raster_info.lineWidth = 1.0;
    raster_info.polygonMode = VK_POLYGON_MODE_FILL;
    raster_info.cullMode = VK_CULL_MODE_BACK_BIT;
    raster_info.frontFace = VK_FRONT_FACE_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisample_info = { VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    multisample_info.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState blendstate = {};
    blendstate.blendEnable = VK_FALSE;
    blendstate.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo blend_info = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    blend_info.attachmentCount = 1;
    blend_info.pAttachments = &blendstate;

    VkDynamicState dynamic_state[] { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamic_info = { VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    dynamic_info.dynamicStateCount = ARRAY_SIZE(dynamic_state);
    dynamic_info.pDynamicStates = dynamic_state;

    VkGraphicsPipelineCreateInfo info = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    info.stageCount = ARRAY_SIZE(shader_stages);
    info.pStages = shader_stages;
    info.pVertexInputState = &vertex_input_info;
    info.pInputAssemblyState = &input_info;
    info.pViewportState = &viewport_info;
    info.pRasterizationState = &raster_info;
    info.pMultisampleState = &multisample_info;
    info.pColorBlendState = &blend_info;
    info.pDynamicState = &dynamic_info;

    info.layout = layout;
    info.renderPass = pass;
    info.subpass = 0;

    VkPipelineCache pipeline_cache = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VK_ASSERT(vkCreateGraphicsPipelines(device, pipeline_cache, 1, &info, nullptr, &pipeline));

    return pipeline;
}

VkSemaphore CreateSemaphore(VkDevice device, VkSemaphoreCreateFlags flags) {
    VkSemaphoreCreateInfo info = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    info.flags = flags;
    VkSemaphore semaphore = VK_NULL_HANDLE;
    VK_ASSERT(vkCreateSemaphore(device, &info, nullptr, &semaphore));
    return semaphore;
}

VkFence CreateFence(VkDevice device, VkFenceCreateFlags flags) {
    VkFenceCreateInfo info { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    info.flags = flags;
    VkFence fence = VK_NULL_HANDLE;
    VK_ASSERT(vkCreateFence(device, &info, nullptr, &fence));
    return fence;
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

Buffer CreateBuffer(VkDevice device, const VkPhysicalDeviceMemoryProperties& properties, const BufferCreateinfo& info)
{
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

void DestroyBuffer(VkDevice device, Buffer* buffer)
{
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
    VkPipelineLayout layout = CreatePipelineLayout(device, { vertex_shader, fragment_shader });
    VkPipeline pipeline = CreatePipeline(device, layout, renderpass, vertex_shader, fragment_shader);

    VkCommandPoolCreateInfo info = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    info.queueFamilyIndex = surface_properties.queue_family_index;

    VkCommandPool command_pool = VK_NULL_HANDLE;
    VK_ASSERT(vkCreateCommandPool(device, &info, nullptr, &command_pool));
    VK_ASSERT(vkResetCommandPool(device, command_pool, 0));

    VkCommandBuffer command_buffer = CreateCommandBuffer(device, command_pool);

    // cpu-gpu synchronize
    VkFence fence = CreateFence(device, 0);

    // gpu-gpu synchronize
    VkSemaphore semaphore = CreateSemaphore(device, 0);
    // VkSemaphore singal_semaphore = CreateSemaphore(device, 0);

#if _DEBUG
    const char* objfile = "models/kitten.obj";
#else
    const char* objfile = "models/buddha.obj";
#endif

    Mesh mesh = LoadMesh(objfile);

    VkMemoryPropertyFlags device_local_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    VkMemoryPropertyFlags host_memory_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
                                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT; 

    const VkDeviceSize vb_size = sizeof(Vertex)*mesh.vertices.size();
    const VkDeviceSize ib_size = sizeof(uint32_t)*mesh.indices.size();

    Buffer staging = CreateBuffer(device, physical_properties.memory, { 
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, host_memory_flags, 1024*1024*64});

    Buffer vb = CreateBuffer(device, physical_properties.memory, {
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        device_local_flags, vb_size });

    Buffer ib = CreateBuffer(device, physical_properties.memory, {
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        device_local_flags, ib_size });

    UploadBuffer(device, command_pool, command_queue, staging, vb, vb_size, mesh.vertices.data());
    UploadBuffer(device, command_pool, command_queue, staging, ib, ib_size, mesh.indices.data());

    VkQueryPool timestamp_pool = CreateQueryPool(device, VK_QUERY_TYPE_TIMESTAMP, 1024, 0);

    uint32_t row_count = 5;
    uint32_t draw_count = row_count*row_count;
    std::vector<MeshDraw> draws(draw_count);
    for (uint32_t i = 0; i < draw_count; i++) {
        draws[i].offset[0] = (float(i % row_count) + 0.5f) / row_count;
        draws[i].offset[1] = (float(i / row_count) + 0.5f) / row_count;
        draws[i].scale[0] = 1.f / row_count;
        draws[i].scale[1] = 1.f / row_count;
    }

    double cpu_average = 0.0, gpu_average = 0.0, wait_average = 0.0;

    while(!glfwWindowShouldClose(windows)) {
        glfwPollEvents();

        auto cpu_begin = std::chrono::steady_clock::now(); 

        uint32_t imageindex = 0;
        VkResult result = vkAcquireNextImageKHR(device, swapchain.swapchain, ~0ull, semaphore, VK_NULL_HANDLE, &imageindex);

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

        VK_ASSERT(vkResetCommandBuffer(command_buffer, 0));
        VkCommandBufferBeginInfo begininfo { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        VK_ASSERT(vkBeginCommandBuffer(command_buffer, &begininfo));

        vkCmdWriteTimestamp(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timestamp_pool, 0);

        VkClearColorValue clear_color = { std::sin(static_cast<float>(glfwGetTime()))*0.5f + 0.5f, 0.5f, 0.5f, 1.0f };
        const VkClearValue clear_values[] = { clear_color, };

        VkRenderPassBeginInfo pass = { VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
        pass.renderPass = renderpass;
        pass.framebuffer = swapchain.framebuffers[imageindex];
        pass.renderArea = scissor;
        pass.clearValueCount = ARRAY_SIZE(clear_values);
        pass.pClearValues = clear_values;

        vkCmdBeginRenderPass(command_buffer, &pass, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        vkCmdSetViewport(command_buffer, 0, 1, &viewport);
        vkCmdSetScissor(command_buffer, 0, 1, &scissor);

    #if FVF
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(command_buffer, 0, 1, &vb.buffer, &offset);
    #else
        VkDescriptorBufferInfo buffer_info = {};
        buffer_info.buffer = vb.buffer;
        buffer_info.offset = 0;
        buffer_info.range = vb.size;

        VkWriteDescriptorSet desc_set { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        desc_set.dstBinding = 0;
        desc_set.descriptorCount = 1;
        desc_set.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        desc_set.pBufferInfo = &buffer_info;

        vkCmdPushDescriptorSetKHR(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &desc_set);
    #endif
        vkCmdBindIndexBuffer(command_buffer, ib.buffer, 0, VK_INDEX_TYPE_UINT32);

        for (auto draw : draws) {
            vkCmdPushConstants(command_buffer, layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(draw), &draw);
            vkCmdDrawIndexed(command_buffer, static_cast<uint32_t>(mesh.indices.size()), 1, 0, 0, 0);
        }
        vkCmdEndRenderPass(command_buffer);

        vkCmdWriteTimestamp(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timestamp_pool, 1);

        VK_ASSERT(vkEndCommandBuffer(command_buffer));

        VkPipelineStageFlags stage_flags[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        VkSubmitInfo submit = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = &semaphore;
        submit.pWaitDstStageMask = stage_flags;
        // submit.signalSemaphoreCount = 1;
        // submit.pSignalSemaphores = &singal_semaphore;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command_buffer;

        VK_ASSERT(vkResetFences(device, 1, &fence));
        VK_ASSERT(vkQueueSubmit(command_queue, 1, &submit, fence));

        VkPresentInfoKHR present_info = { VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
        present_info.swapchainCount = 1;
        present_info.pSwapchains = &swapchain.swapchain;
        present_info.pImageIndices = &imageindex;
        // present_info.pWaitSemaphores = &singal_semaphore;
        // present_info.waitSemaphoreCount = 1;
        vkQueuePresentKHR(present_queue, &present_info);

        auto cpu_end = std::chrono::steady_clock::now(); 

        // 'VK_QUERY_RESULT_WAIT_BIT' seems not working
        auto wait_begin = std::chrono::steady_clock::now();
        // VK_ASSERT(vkWaitForFences(device, 1, &fence, TRUE, ~0ull));
        VK_ASSERT(vkQueueWaitIdle(command_queue));
        auto wait_end = std::chrono::steady_clock::now();

        uint64_t timestamps[2] = {};
        vkGetQueryPoolResults(device, timestamp_pool, 0, 2, sizeof(timestamps), timestamps,
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
    }

    VK_ASSERT(vkDeviceWaitIdle(device));
    vkDestroyFence(device, fence, nullptr);
    vkDestroySemaphore(device, semaphore, nullptr);

    DestroyBuffer(device, &staging);
    DestroyBuffer(device, &vb);
    DestroyBuffer(device, &ib);

    vkDestroyQueryPool(device, timestamp_pool, nullptr);

    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, layout, nullptr);
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