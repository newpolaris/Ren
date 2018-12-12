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
#include <random>

#include "device.h"
#include "mesh.h"
#include "macro.h"
#include "shaders.h"
#include "synchronizes.h"
#include "resources.h"
#include "swapchain.h"
#include "math.h"
#include "bits.h"

#define SUPPORT_MULTIFRAME_IN_FLIGHT 0

namespace {
    bool cluster_culling = true;
};

struct alignas(16) GraphicsData {
    mat4x4 project;
};

// per element memory is allocated align in 16 byte
// but, it seems that there's no neeed to add last element pad
struct alignas(16) CullingData {
    vec4 frustums[6];
    uint32_t draw_count;
};

VkRenderPass CreateRenderPass(VkDevice device, VkFormat color_format, VkFormat depth_format) {
    VkAttachmentDescription desc[2] = {};
    desc[0].format = color_format;
    desc[0].samples = VK_SAMPLE_COUNT_1_BIT;
    desc[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    desc[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    desc[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    desc[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    desc[0].initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    desc[0].finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    desc[1].format = depth_format;
    desc[1].samples = VK_SAMPLE_COUNT_1_BIT;
    desc[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    desc[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    desc[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    desc[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    desc[1].initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    desc[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference color = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
    VkAttachmentReference depth = { 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color;
    subpass.pDepthStencilAttachment = &depth;

    VkRenderPassCreateInfo info = { VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO };
    info.attachmentCount = ARRAY_SIZE(desc);
    info.pAttachments = desc;
    info.subpassCount = 1;
    info.pSubpasses = &subpass;

    VkRenderPass renderpass = VK_NULL_HANDLE;
    VK_ASSERT(vkCreateRenderPass(device, &info, nullptr, &renderpass));
    return renderpass;
}

VkFramebuffer CreateFramebuffer(VkDevice device, VkRenderPass pass, ImageViewList views, VkExtent2D extent) {
    VkFramebufferCreateInfo info = { VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
    info.renderPass = pass;
    info.attachmentCount = static_cast<uint32_t>(views.size());
    info.pAttachments = views.data();
    info.width = extent.width;
    info.height = extent.height;
    info.layers = 1;

    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    VK_ASSERT(vkCreateFramebuffer(device, &info, nullptr, &framebuffer));
    return framebuffer;
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
std::vector<VkDrawIndexedIndirectCommand> CreateIndirectCommandBuffer(const Mesh& mesh,
                                                                      const std::vector<MeshDraw>& draws) {
    std::vector<VkDrawIndexedIndirectCommand> commands;
    for (size_t i = 0; i < draws.size(); i++) {
        for (size_t k = 0; k < mesh.meshlet_instances.size(); k++) {
            auto cone = mesh.meshlets[k].cone;
            auto cosangle = glm::dot(vec3(cone[0], cone[1], cone[2]), vec3(0, 0, 1));
            if (cone[3] < cosangle)
                continue;

            VkDrawIndexedIndirectCommand indirectCmd {};
            indirectCmd.instanceCount = 1;
            indirectCmd.firstInstance = i;
            indirectCmd.firstIndex = mesh.meshlet_instances[k].first;
            indirectCmd.indexCount = mesh.meshlet_instances[k].second;
            indirectCmd.vertexOffset = 0;
            
            commands.push_back(indirectCmd);
        }
    }
    return std::move(commands);
}

/*
 *  @param[in] key The [keyboard key](@ref keys) that was pressed or released.
 *  @param[in] scancode The system-specific scancode of the key.
 *  @param[in] action 'GLFW_PRESS', 'GLFW_RELEASE' or 'GLFW_REPEAT'.
 *  @param[in] mods Bit field describing which [modifier keys](@ref mods) were GLFWkeyfun
 */
static void KeyboardCallback(GLFWwindow* windows, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_C && action == GLFW_PRESS)
        cluster_culling = !cluster_culling;
}

static void KeyboardUpdate(GLFWwindow* windows) {
    if (GLFW_PRESS == glfwGetKey(windows, GLFW_KEY_C))
        cluster_culling = !cluster_culling;
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
    glfwSetKeyCallback(windows, KeyboardCallback);

    VkSurfaceKHR surface = CreateSurface(instance, windows);

    VkPhysicalDevice physical_device = CreatePhysicalDevice(instance);

    const auto physical_properties = CreatePhysicalDeviceProperties(physical_device);
    const auto surface_properties = CreateSurfaceProperties(physical_device, physical_properties.queue, surface);

    VkDevice device = CreateDevice(physical_device, surface_properties.queue_family_index);

    VkSurfaceCapabilitiesKHR capabilities = {};
    VK_ASSERT(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &capabilities));

    VkFormat color_format = surface_properties.format.format;
    VkFormat depth_format = VK_FORMAT_D32_SFLOAT;

    const VkRenderPass mesh_renderpass = CreateRenderPass(device, surface_properties.format.format, VK_FORMAT_D32_SFLOAT);

    Swapchain swapchain = {};
    VkViewport viewport = {};
    VkRect2D scissor = {};

    UpdateViewportScissor(swapchain.extent, &viewport, &scissor); 

    constexpr uint32_t queue_index = 0;
    VkQueue command_queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, surface_properties.queue_family_index, queue_index, &command_queue);
    VkQueue present_queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, surface_properties.queue_family_index, queue_index, &present_queue);

    ShaderModule drawcmd_shader = CreateShaderModule(device, "shaders/05.rin/drawcmd.comp.spv");
    Program drawcmd_program = CreateProgram(device, VK_PIPELINE_BIND_POINT_COMPUTE, {drawcmd_shader});
    VkPipeline drawcmd_pipeline = CreateComputePipeline(device, drawcmd_program.layout, drawcmd_shader);

    ShaderModule vertex_shader = CreateShaderModule(device, "shaders/05.rin/base.vert.spv");
    ShaderModule fragment_shader = CreateShaderModule(device, "shaders/05.rin/base.frag.spv");
    ShaderModules shaders = { vertex_shader, fragment_shader };

    Program mesh_program = CreateProgram(device, VK_PIPELINE_BIND_POINT_GRAPHICS, shaders);
    VkPipeline mesh_pipeline = CreateGraphicsPipeline(device, mesh_program.layout, mesh_renderpass, shaders);

    VkCommandPoolCreateInfo info = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    info.queueFamilyIndex = surface_properties.queue_family_index;

    VkCommandPool command_pool = VK_NULL_HANDLE;
    VK_ASSERT(vkCreateCommandPool(device, &info, nullptr, &command_pool));
    VK_ASSERT(vkResetCommandPool(device, command_pool, 0));

    const size_t chains = std::min(capabilities.minImageCount + 1, capabilities.maxImageCount);

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

    VkMemoryPropertyFlags device_local_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    VkMemoryPropertyFlags host_memory_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
                                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT; 

    const VkDeviceSize vb_size = sizeof(Vertex)*mesh.vertices.size();
    const VkDeviceSize ib_size = sizeof(uint32_t)*mesh.indices.size();
    const VkDeviceSize mib_size = mesh.meshlet_indices.size() * sizeof(uint32_t);

    Buffer staging = CreateBuffer(device, physical_properties.memory, { 
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, host_memory_flags, 1024*1024*64});

    Buffer vb = CreateBuffer(device, physical_properties.memory, {
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        device_local_flags, vb_size });

    Buffer ib = CreateBuffer(device, physical_properties.memory, {
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        device_local_flags, ib_size });

    Buffer meshlet_index_buffer = CreateBuffer(device, physical_properties.memory, {
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        device_local_flags, mib_size });

    UploadBuffer(device, command_pool, command_queue, staging, vb, vb_size, mesh.vertices.data());
    UploadBuffer(device, command_pool, command_queue, staging, ib, ib_size, mesh.indices.data());
    UploadBuffer(device, command_pool, command_queue, staging, meshlet_index_buffer, mib_size, mesh.meshlet_indices.data());

    VkQueryPool timestamp_pool = CreateQueryPool(device, VK_QUERY_TYPE_TIMESTAMP, 1024, 0);

    std::default_random_engine eng {10};
    std::uniform_real_distribution<float> urd(0, 1);
    const uint32_t draw_count = 2000;
    std::vector<MeshDraw> draws(draw_count);
    for (uint32_t i = 0; i < draw_count; i++) {
        vec3 axis = vec3( urd(eng)*2 - 1, urd(eng)*2 - 1, urd(eng)*2 - 1);
        float angle = glm::radians(urd(eng) * 90.f);

        draws[i].position[0] = urd(eng) * 20.f - 10.f;
        draws[i].position[1] = urd(eng) * 20.f - 10.f;
        draws[i].position[2] = urd(eng) * 20.f - 10.f;
        draws[i].scale = urd(eng) + 0.5f;
        draws[i].orientation = glm::rotate(quat(1, 0, 0, 0), angle, axis); 
        draws[i].index_count = static_cast<uint32_t>(mesh.indices.size());
        draws[i].center = mesh.center;
        draws[i].radius = mesh.radius;
    }

    // meshdarw buffer
    const VkDeviceSize meshdrawbuffer_size = sizeof(MeshDraw) * draws.size();
    Buffer meshdraw_buffer = CreateBuffer(device, physical_properties.memory, {
        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        device_local_flags, meshdrawbuffer_size });
    UploadBuffer(device, command_pool, command_queue, staging, meshdraw_buffer, meshdrawbuffer_size, draws.data());

    // meshdraw-command buffer
    const VkDeviceSize meshdrawcommandbuffer_size = 1024*1024*128;
    Buffer meshdraw_command_buffer = CreateBuffer(device, physical_properties.memory, {
        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        device_local_flags, meshdrawcommandbuffer_size });

    Buffer draw_count_buffer = CreateBuffer(device, physical_properties.memory, {
        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        device_local_flags, sizeof(uint32_t)});

    std::vector<VkDrawIndexedIndirectCommand> indirects = CreateIndirectCommandBuffer(mesh, draws);
    const uint32_t indirect_draw_count = static_cast<uint32_t>(indirects.size());
    const VkDeviceSize idcb_size = indirects.size() * sizeof(VkDrawIndexedIndirectCommand);

    // muliple indirect draw buffer for drawing culled meshlet cluster
    Buffer idcb = CreateBuffer(device, physical_properties.memory, {
        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        device_local_flags, idcb_size });

    UploadBuffer(device, command_pool, command_queue, staging, idcb, idcb_size, indirects.data());

    double cpu_average = 0.0, gpu_average = 0.0, wait_average = 0.0;

    size_t current_frame = 0;

    Image color = {};
    Image depth = {};
    VkFramebuffer framebuffer = VK_NULL_HANDLE;

    while(!glfwWindowShouldClose(windows)) {
        glfwPollEvents();
        // KeyboardUpdate(windows);

        auto fence = fences[current_frame];
    #if SUPPORT_MULTIFRAME_IN_FLIGHT
        VK_ASSERT(vkWaitForFences(device, 1, &fence, TRUE, ~0ull));
    #endif

        auto cpu_begin = std::chrono::steady_clock::now(); 
        auto semaphore = semaphores[current_frame];
        auto signal_semaphore = signal_semaphores[current_frame];

        uint32_t image_index = 0;
        VkResult result = VK_ERROR_OUT_OF_DATE_KHR;
        // to skip signals smaphore 
        if (swapchain.swapchain)
            result = vkAcquireNextImageKHR(device, swapchain.swapchain, ~0ull, semaphore, VK_NULL_HANDLE, &image_index);
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
            VK_ASSERT(vkDeviceWaitIdle(device));
            VK_ASSERT(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &capabilities));
            if (capabilities.currentExtent.height == 0 && capabilities.currentExtent.width == 0)
                continue;
            CreateSwapchain(device, capabilities, surface, surface_properties, mesh_renderpass, &swapchain);
            UpdateViewportScissor(swapchain.extent, &viewport, &scissor);

            vkDestroyFramebuffer(device, framebuffer, nullptr);
            DestroyImage(device, &color);
            DestroyImage(device, &depth);

            const VkExtent2D extent = { capabilities.currentExtent.width, capabilities.currentExtent.height };
            color = CreateImage(device, physical_properties.memory, {
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                color_format,
                extent
            });

            depth = CreateImage(device, physical_properties.memory, {
                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                depth_format,
                extent
            });
            framebuffer = CreateFramebuffer(device, mesh_renderpass, {color.view, depth.view}, extent);
            continue;
        } 
        ASSERT(result == VK_SUCCESS);

        auto command_buffer = command_buffers[image_index];

        VK_ASSERT(vkResetCommandBuffer(command_buffer, 0));
        VkCommandBufferBeginInfo begininfo { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        VK_ASSERT(vkBeginCommandBuffer(command_buffer, &begininfo));

        vkCmdWriteTimestamp(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timestamp_pool, current_frame*2);

        uint32_t indirect_command_count = draw_count;
        if (cluster_culling)
            indirect_command_count = indirect_draw_count;

        float aspect = static_cast<float>(swapchain.extent.width) / swapchain.extent.height;
        const mat4x4 project = PerspectiveProjection(glm::radians(70.f), aspect, 0.01f);

        const GraphicsData graphics_data = { project };
        CullingData culling_data;
        culling_data.draw_count = indirect_command_count;

        const float max_draw_distance = 200.f;
        const auto frustums = GetFrustum(project, max_draw_distance);
        for (uint32_t i = 0; i < 6; i++)
            culling_data.frustums[i] = frustums[i];

        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, drawcmd_pipeline);
        if (!cluster_culling)
        {
            vkCmdFillBuffer(command_buffer, draw_count_buffer.buffer, 0, 4, 0);

            auto fill_barrier = CreateBufferBarrier(draw_count_buffer, VK_ACCESS_TRANSFER_WRITE_BIT,
                                                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
            vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, 
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, 0, 1, &fill_barrier, 0, 0);

            // input, output, output
            PushDescriptorSets descriptors[] = { meshdraw_buffer, meshdraw_command_buffer, draw_count_buffer };
            vkCmdPushDescriptorSetWithTemplateKHR(command_buffer, drawcmd_program.update, drawcmd_program.layout, 0, &descriptors);

            vkCmdPushConstants(command_buffer, drawcmd_program.layout, drawcmd_program.push_constant_stages, 0, 
                               sizeof(culling_data), &culling_data);
            vkCmdDispatch(command_buffer, uint32_t((draw_count + 31) / 32), 1, 1);

            // Without barrier, first attempt that change culling status will result empty screen in few frames;
            VkBufferMemoryBarrier command_end[] = { 
                CreateBufferBarrier(meshdraw_command_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT),
                CreateBufferBarrier(draw_count_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT),
            };
			vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                                 0, 0, 0, ARRAY_SIZE(command_end), command_end, 0, 0);
        }

        VkImageMemoryBarrier render_begin[] = {
            CreateImageBarrier(color.image, 0, 0, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT),
            CreateImageBarrier(depth.image, 0, 0, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT),
        };
        vkCmdPipelineBarrier(command_buffer,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_DEPENDENCY_BY_REGION_BIT, 0, nullptr, 0, nullptr,
            ARRAY_SIZE(render_begin), render_begin);

        VkClearColorValue clear_color = { std::sin(static_cast<float>(glfwGetTime()))*0.5f + 0.5f, 0.5f, 0.5f, 1.0f };
        VkClearDepthStencilValue clear_depth = { 0.0f, 0 };
        VkClearValue clear_values[2];
        clear_values[0].color = clear_color;
        clear_values[1].depthStencil = clear_depth;

        VkRenderPassBeginInfo pass = { VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
        pass.renderPass = mesh_renderpass;
        pass.framebuffer = framebuffer;
        pass.renderArea = scissor;
        pass.clearValueCount = ARRAY_SIZE(clear_values);
        pass.pClearValues = clear_values;

        vkCmdBeginRenderPass(command_buffer, &pass, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, mesh_pipeline);
        vkCmdSetViewport(command_buffer, 0, 1, &viewport);
        vkCmdSetScissor(command_buffer, 0, 1, &scissor);

        if (cluster_culling) {
            PushDescriptorSets descriptors[] = {vb, meshdraw_buffer, idcb};
            vkCmdPushDescriptorSetWithTemplateKHR(command_buffer, mesh_program.update, mesh_program.layout, 0, &descriptors);
        } else {
            PushDescriptorSets descriptors[] = {vb, meshdraw_buffer, meshdraw_command_buffer};
            vkCmdPushDescriptorSetWithTemplateKHR(command_buffer, mesh_program.update, mesh_program.layout, 0, &descriptors);
        }

        if (cluster_culling)
            vkCmdBindIndexBuffer(command_buffer, meshlet_index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);
        else
            vkCmdBindIndexBuffer(command_buffer, ib.buffer, 0, VK_INDEX_TYPE_UINT32);

        vkCmdPushConstants(command_buffer, mesh_program.layout, mesh_program.push_constant_stages, 0, sizeof(graphics_data), &graphics_data);
        if (cluster_culling)
            vkCmdDrawIndexedIndirect(command_buffer, idcb.buffer, 0, indirect_command_count, sizeof(VkDrawIndexedIndirectCommand));
        else 
            vkCmdDrawIndexedIndirectCountKHR(command_buffer, meshdraw_command_buffer.buffer, 0, draw_count_buffer.buffer, 0, 
                                             indirect_command_count, sizeof(MeshDrawCommand));

        vkCmdEndRenderPass(command_buffer);

        VkImageMemoryBarrier copyBarriers[] = {
            CreateImageBarrier(swapchain.images[image_index], 0, VK_ACCESS_TRANSFER_WRITE_BIT, 
                               VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT),
        };

        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_DEPENDENCY_BY_REGION_BIT, 0, nullptr, 0, nullptr, ARRAYSIZE(copyBarriers), copyBarriers);

        // TODO: implement scale paste to swapchain
        VkImageCopy copy_region = {};
        copy_region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy_region.srcSubresource.layerCount = 1;
        copy_region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy_region.dstSubresource.layerCount = 1;
        copy_region.extent = { swapchain.extent.width, swapchain.extent.height };

        vkCmdCopyImage(command_buffer, color.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, swapchain.images[image_index],
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_region);

        VkImageMemoryBarrier present = CreateImageBarrier(swapchain.images[image_index], 
                                                          VK_ACCESS_TRANSFER_WRITE_BIT, 0, 
                                                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                                                          VK_IMAGE_ASPECT_COLOR_BIT);

        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                             VK_DEPENDENCY_BY_REGION_BIT, 0, nullptr, 0, nullptr, 1, &present);

        vkCmdWriteTimestamp(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timestamp_pool, current_frame*2+1);

        VK_ASSERT(vkEndCommandBuffer(command_buffer));

        VkPipelineStageFlags stage_flags[] = { VK_PIPELINE_STAGE_TRANSFER_BIT };
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
    DestroyBuffer(device, &meshlet_index_buffer);
    DestroyBuffer(device, &meshdraw_buffer);
    DestroyBuffer(device, &idcb);
    DestroyBuffer(device, &meshdraw_command_buffer);
    DestroyBuffer(device, &draw_count_buffer);

    vkDestroyFramebuffer(device, framebuffer, nullptr);

    DestroyImage(device, &color);
    DestroyImage(device, &depth);
    
    vkDestroyQueryPool(device, timestamp_pool, nullptr);

    vkDestroyPipeline(device, drawcmd_pipeline, nullptr);
    DestroyProgram(device, &drawcmd_program);

    vkDestroyShaderModule(device, drawcmd_shader.module, nullptr);

    vkDestroyPipeline(device, mesh_pipeline, nullptr);
    DestroyProgram(device, &mesh_program);

    vkDestroyShaderModule(device, vertex_shader.module, nullptr);
    vkDestroyShaderModule(device, fragment_shader.module, nullptr);

    vkDestroyCommandPool(device, command_pool, nullptr);

    DestroySwapchain(device, &swapchain);

    vkDestroyRenderPass(device, mesh_renderpass, nullptr);
    vkDestroyDevice(device, nullptr);
    device = VK_NULL_HANDLE;

    DestroyDebugCallback(instance, messenger);

    vkDestroyInstance(instance, nullptr);
    instance = VK_NULL_HANDLE;

    glfwDestroyWindow(windows);
    windows = VK_NULL_HANDLE;

    return 0;
}