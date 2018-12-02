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
#include <fstream>
#include <objparser.h>

#ifndef ASSERT
#define ASSERT(x) \
    assert((x))
#endif

#ifndef VK_ASSERT
#define VK_ASSERT(x) \
    do { \
        VkResult result = x; \
        ASSERT(VK_SUCCESS == result); \
    } while(0)
#endif

#ifndef ARRAY_SIZE
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#endif

std::vector<char> readfile(const std::string& filename)
{
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    ASSERT(file.is_open());

    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    return buffer;
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

// Choose graphics queue that also support present
uint32_t GetQueueFamilyIndex(VkPhysicalDevice device, VkSurfaceKHR surface) {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);

    std::vector<VkQueueFamilyProperties> properties(count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, properties.data());

    for (size_t i = 0; i < properties.size(); i++) {
        VkBool32 support = VK_FALSE;
        VK_ASSERT(vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &support));

        if ((properties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && properties[i].queueCount > 0)
            return i;
    }
    return VK_QUEUE_FAMILY_IGNORED;
}

VkDevice CreateDevice(VkPhysicalDevice physical_device, VkSurfaceKHR surface, uint32_t queue_family_index) {
    std::vector<const char*> device_extensions {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    };

    const float qeue_priorites[] = { 1.f };
    VkDeviceQueueCreateInfo queue_create_info = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO  };
    queue_create_info.queueFamilyIndex = queue_family_index;
    queue_create_info.queueCount = ARRAY_SIZE(qeue_priorites);
    queue_create_info.pQueuePriorities = qeue_priorites;

    VkPhysicalDeviceFeatures features = {};
    vkGetPhysicalDeviceFeatures(physical_device, &features);

    VkDeviceCreateInfo info = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    info.queueCreateInfoCount = 1;
    info.pQueueCreateInfos = &queue_create_info;
    info.enabledExtensionCount = static_cast<uint32_t>(device_extensions.size());;
    info.ppEnabledExtensionNames = device_extensions.data();
    info.pEnabledFeatures = &features;

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

SurfaceProperties CreateSurfaceProperties(VkPhysicalDevice physical_device, VkSurfaceKHR surface) {
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

    const uint32_t queue_family_index = GetQueueFamilyIndex(physical_device, surface);
    ASSERT(queue_family_index != VK_QUEUE_FAMILY_IGNORED);

    return { swapchain_format, present_mode, queue_family_index };
}

VkSwapchainKHR CreateSwapchain(VkDevice device, VkSurfaceKHR surface, SurfaceProperties properties,
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
    if (!result)
        return;

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
                     SurfaceProperties properties, VkRenderPass renderpass, Swapchain* result) {
    if (!result)
        return;

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

VkShaderModule CreateShaderModule(VkDevice device, const char* filepath) {
    auto code = readfile(filepath);
    VkShaderModuleCreateInfo info = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    info.codeSize = code.size();
    info.pCode = reinterpret_cast<uint32_t*>(code.data());

    VkShaderModule module = VK_NULL_HANDLE;
    VK_ASSERT(vkCreateShaderModule(device, &info, nullptr, &module));
    return module;
}

VkPipelineLayout CreatePipelineLayout(VkDevice device) {
    VkPipelineLayoutCreateInfo info = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VK_ASSERT(vkCreatePipelineLayout(device, &info, nullptr, &layout));
    return layout;
}

VkPipeline CreatePipeline(VkDevice device, VkPipelineLayout layout, VkRenderPass pass, VkShaderModule vs, VkShaderModule fs) { 
    VkPipelineShaderStageCreateInfo vertex_info = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    vertex_info.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertex_info.module = vs;
    vertex_info.pName = "main";

    VkPipelineShaderStageCreateInfo fragment_info = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    fragment_info.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragment_info.module = fs;
    fragment_info.pName = "main";
    VkPipelineShaderStageCreateInfo shader_stages[] = { vertex_info, fragment_info };

    VkPipelineVertexInputStateCreateInfo vertexinputinfo { VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };

    VkPipelineInputAssemblyStateCreateInfo inputinfo { VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    inputinfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewport_info { VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    viewport_info.viewportCount = 1;
    viewport_info.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo raster_info = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    raster_info.polygonMode = VK_POLYGON_MODE_FILL;
    raster_info.cullMode = VK_CULL_MODE_NONE;
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

    VkGraphicsPipelineCreateInfo pipeline_info = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    pipeline_info.stageCount = ARRAY_SIZE(shader_stages);
    pipeline_info.pStages = shader_stages;
    pipeline_info.pVertexInputState = &vertexinputinfo;
    pipeline_info.pInputAssemblyState = &inputinfo;
    pipeline_info.pViewportState = &viewport_info;
    pipeline_info.pRasterizationState = &raster_info;
    pipeline_info.pMultisampleState = &multisample_info;
    pipeline_info.pColorBlendState = &blend_info;
    pipeline_info.pDynamicState = &dynamic_info;

    pipeline_info.layout = layout;
    pipeline_info.renderPass = pass;
    pipeline_info.subpass = 0;

    VkPipelineCache pipeline_cache = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VK_ASSERT(vkCreateGraphicsPipelines(device, pipeline_cache, 1, &pipeline_info, nullptr, &pipeline));

    return pipeline;
}

VkSemaphore CreateSemaphore(VkDevice device, VkSemaphoreCreateFlags flags) {
    VkSemaphoreCreateInfo info = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    info.flags = flags;
    VkSemaphore semaphore = VK_NULL_HANDLE;
    VK_ASSERT(vkCreateSemaphore(device, &info, nullptr, &semaphore));
    return semaphore;
}

VkFence CreateFence(VkDevice device, VkFenceCreateFlagBits flags) {
    VkFenceCreateInfo info { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    VkFence fence = VK_NULL_HANDLE;
    VK_ASSERT(vkCreateFence(device, &info, nullptr, &fence));
    return fence;
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
    const char* application_name = "Hello Triangle";
    const char* engine_name = "Rin";

    if (glfwInit() != GLFW_TRUE)
        return EXIT_FAILURE;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    volkInitialize();

    VkInstance instance = CreateInstance(application_name, engine_name);
    if (instance == VK_NULL_HANDLE)
        return EXIT_FAILURE;

    volkLoadInstance(instance);

    VkDebugUtilsMessengerEXT messenger = CreateDebugCallback(instance, DebugCallback);

    GLFWwindow* windows = glfwCreateWindow(1024, 768, application_name, nullptr, nullptr);
    VkSurfaceKHR surface = CreateSurface(instance, windows);

    VkPhysicalDevice physical_device = CreatePhysicalDevice(instance);

    const SurfaceProperties properties = CreateSurfaceProperties(physical_device, surface);
    VkDevice device = CreateDevice(physical_device, surface, properties.queue_family_index);

    VkSurfaceCapabilitiesKHR capabilities = {};
    VK_ASSERT(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &capabilities));
    const VkRenderPass renderpass = CreateRenderPass(device, properties.format.format);

    Swapchain swapchain = {};
    CreateSwapchain(device, capabilities, surface, properties, renderpass, &swapchain);
    
    VkViewport viewport = {};
    VkRect2D scissor = {};

    UpdateViewportScissor(swapchain.extent, &viewport, &scissor); 

    constexpr uint32_t queue_index = 0;
    VkQueue command_queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, properties.queue_family_index, queue_index, &command_queue);
    VkQueue present_queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, properties.queue_family_index, queue_index, &present_queue);

    VkShaderModule vertex_module = CreateShaderModule(device, "shaders/05.rin/base.vert.spv");
    VkShaderModule fragment_module = CreateShaderModule(device, "shaders/05.rin/base.frag.spv");
    VkPipelineLayout layout = CreatePipelineLayout(device);
    VkPipeline pipeline = CreatePipeline(device, layout, renderpass, vertex_module, fragment_module);

    VkCommandPoolCreateInfo info = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    info.queueFamilyIndex = properties.queue_family_index;

    VkCommandPool commandpool = VK_NULL_HANDLE;
    VK_ASSERT(vkCreateCommandPool(device, &info, nullptr, &commandpool));
    VK_ASSERT(vkResetCommandPool(device, commandpool, 0));

    VkCommandBufferAllocateInfo allocate_info { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    allocate_info.commandPool = commandpool;
    allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate_info.commandBufferCount = 1;

    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    VK_ASSERT(vkAllocateCommandBuffers(device, &allocate_info, &command_buffer));

    // cpu-gpu synchronize
    VkFence fence = CreateFence(device, VK_FENCE_CREATE_SIGNALED_BIT);

    // gpu-gpu synchronize
    VkSemaphore semaphore = CreateSemaphore(device, 0);

    while (!glfwWindowShouldClose(windows)) {
        glfwPollEvents();

        VK_ASSERT(vkWaitForFences(device, 1, &fence, TRUE, ~0ull));

        uint32_t imageindex = 0;
        VkResult result = vkAcquireNextImageKHR(device, swapchain.swapchain, ~0ull, semaphore, VK_NULL_HANDLE, &imageindex);

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
            VK_ASSERT(vkDeviceWaitIdle(device));
            VK_ASSERT(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &capabilities));
            if (capabilities.currentExtent.height == 0 && capabilities.currentExtent.width == 0)
                continue;
            CreateSwapchain(device, capabilities, surface, properties, renderpass, &swapchain);
            UpdateViewportScissor(swapchain.extent, &viewport, &scissor);
            continue;
        } 
        ASSERT(result == VK_SUCCESS);

        VK_ASSERT(vkResetCommandBuffer(command_buffer, 0));
        VkCommandBufferBeginInfo begininfo { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        VK_ASSERT(vkBeginCommandBuffer(command_buffer, &begininfo));

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
        vkCmdDraw(command_buffer, 3, 1, 0, 0);
        vkCmdEndRenderPass(command_buffer);
        VK_ASSERT(vkEndCommandBuffer(command_buffer));

        VkPipelineStageFlags stage_flags[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        VkSubmitInfo submit = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = &semaphore;
        submit.pWaitDstStageMask = stage_flags;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command_buffer;

        VK_ASSERT(vkResetFences(device, 1, &fence));
        VK_ASSERT(vkQueueSubmit(command_queue, 1, &submit, fence));

        VkPresentInfoKHR present_info = { VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
        present_info.swapchainCount = 1;
        present_info.pSwapchains = &swapchain.swapchain;
        present_info.pImageIndices = &imageindex;
        vkQueuePresentKHR(present_queue, &present_info);
    }

    VK_ASSERT(vkDeviceWaitIdle(device));
    vkDestroyFence(device, fence, nullptr);
    vkDestroySemaphore(device, semaphore, nullptr);

    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, layout, nullptr);
    vkDestroyShaderModule(device, vertex_module, nullptr);
    vkDestroyShaderModule(device, fragment_module, nullptr);

    vkDestroyCommandPool(device, commandpool, nullptr);

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