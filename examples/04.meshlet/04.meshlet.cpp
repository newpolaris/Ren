#ifdef _WIN32
#include <windows.h>
#endif

#include "common.h"

#define GLFW_INCLUDE_VULKAN
#include <glfw/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <random>
#include <chrono>
#include <array>
#include <set>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <functional>
#include <sstream>

#include <objparser.h>
#include <meshoptimizer.h>
#include "shaders.h"
#include "filesystem.h"
#include "pipeline.h"
#include "swapchain.h"

const int WIDTH = 1280;
const int HEIGHT = 960;
const int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<const char*> validationLayers = {
    "VK_LAYER_LUNARG_standard_validation"
};

const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
    VK_KHR_16BIT_STORAGE_EXTENSION_NAME,
    VK_KHR_8BIT_STORAGE_EXTENSION_NAME,
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct Vertex {
    uint16_t vx, vy, vz, vw;
    uint8_t nx, ny, nz, nw;
    uint16_t tu, tv;
};

enum { kMeshletVertices = 64 };
enum { kMeshletTri = 126 };

struct alignas(16) Meshlet
{
    float cone[4];
    uint32_t vertices[kMeshletVertices];
    uint8_t indices[kMeshletTri*3]; // up to 126 triangles
    uint8_t triangleCount;
    uint8_t vertexCount;
};

struct Mesh
{
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<Meshlet> meshlets;
    std::vector<uint32_t> meshletIndices;
    std::vector<std::pair<uint32_t, uint32_t>> meshletInstances;
};

float halfToFloat(uint16_t v)
{
    uint16_t sign = v >> 15;
    uint16_t exp = (v >> 10) & 31;
    uint16_t man = v & 1023;

    assert(exp != 31);
    if (exp == 0) 
    {
        assert(man == 0);
        return 0.f;
    }
    return (sign == 0 ? 1.f : -1.f) * ldexpf(float(man + 1024)/1024.f, exp - 15);
}

bool loadMesh(Mesh& result, const std::string& path)
{
    ObjFile file;
    if (objParseFile(file, path.c_str()))
    {
        size_t index_count = file.f_size / 3;

        std::vector<Vertex> vertices(index_count);

        for (size_t i = 0; i < index_count; i++)
        {
            Vertex& v = vertices[i];

            int vi = file.f[i * 3 + 0];
            int vti = file.f[i * 3 + 1];
            int vni = file.f[i * 3 + 2];

            float nx = vni < 0 ? 0.f : file.vn[vni * 3 + 0];
            float ny = vni < 0 ? 0.f : file.vn[vni * 3 + 1];
            float nz = vni < 0 ? 1.f : file.vn[vni * 3 + 2];

            v.vx = meshopt_quantizeHalf(file.v[vi * 3 + 0]);
            v.vy = meshopt_quantizeHalf(file.v[vi * 3 + 1]);
            v.vz = meshopt_quantizeHalf(file.v[vi * 3 + 2]);
            v.vw = 0;
            v.nx = uint8_t(nx * 127.f + 127.f); // TODO: fix rounding
            v.ny = uint8_t(ny * 127.f + 127.f); // TODO: fix rounding
            v.nz = uint8_t(nz * 127.f + 127.f); // TODO: fix rounding
            v.tu = meshopt_quantizeHalf(vti < 0 ? 0.f : file.vt[vti * 3 + 0]);
            v.tv = meshopt_quantizeHalf(vti < 0 ? 0.f : file.vt[vti * 3 + 2]);
        }

        if (false)
        {
            result.vertices = vertices;
            result.indices.resize(index_count);

            for (size_t i = 0; i < index_count; i++)
                result.indices[i] = uint32_t(i);
        }
        else
        {
            std::vector<uint32_t> remap(index_count);
            size_t vertex_count = meshopt_generateVertexRemap(remap.data(), 0, index_count, vertices.data(), index_count, sizeof(Vertex));

            result.vertices.resize(vertex_count);
            result.indices.resize(index_count);

            meshopt_remapVertexBuffer(result.vertices.data(), vertices.data(), index_count, sizeof(Vertex), remap.data());
            meshopt_remapIndexBuffer(result.indices.data(), 0, index_count, remap.data());

            meshopt_optimizeVertexCache(result.indices.data(), result.indices.data(), index_count, vertex_count);
            meshopt_optimizeVertexFetch(result.vertices.data(), result.indices.data(), index_count, result.vertices.data(), vertex_count, sizeof(Vertex));
        }
        return true;
    }
    throw std::runtime_error("failed to load model!");
}

void buildMeshlets(Mesh& mesh)
{
    Meshlet meshlet = {};
    std::vector<uint8_t> meshletVertices(mesh.vertices.size(), 0xff);

    for (size_t i = 0; i < mesh.indices.size(); i+= 3)
    {
        unsigned int a = mesh.indices[i + 0];
        unsigned int b = mesh.indices[i + 1];
        unsigned int c = mesh.indices[i + 2];

        auto& av = meshletVertices[a];
        auto& bv = meshletVertices[b];
        auto& cv = meshletVertices[c];

        if (meshlet.vertexCount + (av == 0xff) + (bv == 0xff) + (cv == 0xff) > kMeshletVertices || meshlet.triangleCount >= kMeshletTri)
        {
            mesh.meshlets.push_back(meshlet);

            for (size_t j = 0; j < meshlet.vertexCount; j++)
                meshletVertices[meshlet.vertices[j]] = 0xff;
            meshlet = {};
        }

        if (av == 0xff)
        {
            av = meshlet.vertexCount;
            meshlet.vertices[meshlet.vertexCount++] = a;
        }

		if (bv == 0xff)
		{
			bv = meshlet.vertexCount;
			meshlet.vertices[meshlet.vertexCount++] = b;
		}

		if (cv == 0xff)
		{
			cv = meshlet.vertexCount;
			meshlet.vertices[meshlet.vertexCount++] = c;
		}

		meshlet.indices[meshlet.triangleCount * 3 + 0] = av;
		meshlet.indices[meshlet.triangleCount * 3 + 1] = bv;
		meshlet.indices[meshlet.triangleCount * 3 + 2] = cv;
        meshlet.triangleCount++;
    }
    if (meshlet.triangleCount)
        mesh.meshlets.push_back(meshlet);
}

void buildMeshletCones(Mesh& mesh)
{
    for (auto& meshlet : mesh.meshlets)
    {
        std::vector<float[3]> normals(kMeshletTri);

        for (uint16_t i = 0; i < meshlet.triangleCount; i++)
        {
            auto a = meshlet.indices[i*3 + 0];
            auto b = meshlet.indices[i*3 + 1];
            auto c = meshlet.indices[i*3 + 2];

            const auto& va = mesh.vertices[meshlet.vertices[a]];
            const auto& vb = mesh.vertices[meshlet.vertices[b]];
            const auto& vc = mesh.vertices[meshlet.vertices[c]];

            float p0[3] = { halfToFloat(va.vx), halfToFloat(va.vy), halfToFloat(va.vz) };
            float p1[3] = { halfToFloat(vb.vx), halfToFloat(vb.vy), halfToFloat(vb.vz) };
            float p2[3] = { halfToFloat(vc.vx), halfToFloat(vc.vy), halfToFloat(vc.vz) };

            float p10[3] = { p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2] };
            float p20[3] = { p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2] };

            // cross(p10, p20)
            float normalx = p10[1]*p20[2] - p10[2]*p20[1];
            float normaly = p10[2]*p20[0] - p10[0]*p20[2];
            float normalz = p10[0]*p20[1] - p10[1]*p20[0];

            float area = sqrtf(normalx*normalx + normaly*normaly + normalz*normalz);
            float invarea = area == 0.f ? 0.f : 1.f / area;

            normals[i][0] = normalx * invarea;
            normals[i][1] = normaly * invarea;
            normals[i][2] = normalz * invarea;
        }

        float normal[4] = {};
        for (int i = 0; i < meshlet.triangleCount; i++)
            for (int t = 0; t < 3; t++)
                normal[t] += normals[i][t];

        for (int t = 0; t < 3; t++)
            normal[t] /= meshlet.triangleCount;

        float length = sqrtf(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]);
        if (length <= 0.f)
        {
            normal[0] = 1.f;
            normal[1] = 0.f;
            normal[2] = 0.f;
            normal[3] = 1.f;
        }
        else
        {
            float inverseLength = 1.f / length;
            for (int t = 0; t < 3; t++)
                normal[t] *= inverseLength;

            float mindp = 1.f;
            for (int i = 0; i < meshlet.triangleCount; i++)
            {
                float dp = 0.f;
                for (int t = 0; t < 3; t++)
                    dp += normals[i][t]*normal[t];
                mindp = glm::min(mindp, dp);
            }
            normal[3] = mindp <= 0.f ? 1 : sqrtf(1 - mindp * mindp);
        }
        for (int t = 0; t < 4; t++)
            meshlet.cone[t] = normal[t];
    }
}

void buildMeshletIndices(Mesh& mesh)
{
    uint32_t cnt = 0;
    std::vector<uint32_t> meshletIndices(mesh.indices.size());
    for (const auto& meshlet : mesh.meshlets)
    {
        uint32_t start = cnt;
        for (uint32_t k = 0; k < uint32_t(meshlet.triangleCount)*3; k++)
            meshletIndices[cnt++] = meshlet.vertices[meshlet.indices[k]];
        mesh.meshletInstances.push_back({ start, cnt - start });
    }
    mesh.meshletIndices = meshletIndices;

    size_t culled = 0;
    for (Meshlet& meshlet : mesh.meshlets)
        if (meshlet.cone[2] > meshlet.cone[3])
            culled++;
#if _DEBUG
    printf("Culled meshlets: %d/%d\n", int(culled), int(mesh.meshlets.size()));
#endif
}

struct Buffer
{
    VkBuffer buffer;
    VkDeviceMemory memory;
    void* data;
    size_t size;
};

struct UniformBufferObject {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};

class HelloTriangleApplication
{
public:

    HelloTriangleApplication()
    {
    }

    void run()
    {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:

    GLFWwindow* window = nullptr;

    bool bMeshShaderSupported = false;
    bool bMehsShaderEnabled = false;

    VkInstance instance;
    VkDebugUtilsMessengerEXT callback;
    VkSurfaceKHR surface;

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;

    VkQueue graphicsQueue;
    VkQueue presentQueue;

    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    std::vector<VkImageView> swapChainImageViews;
    std::vector<VkFramebuffer> swapChainFramebuffers;

    VkRenderPass renderPass;
    VkPipelineLayout meshPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout meshletPipelineLayout = VK_NULL_HANDLE;
    VkDescriptorUpdateTemplate meshUpdateTemplate = VK_NULL_HANDLE;
    VkDescriptorUpdateTemplate meshletUpdateTemplate = VK_NULL_HANDLE;
    VkPipeline meshPipeline = VK_NULL_HANDLE;
    VkPipeline meshletPipeline = VK_NULL_HANDLE;

    VkCommandPool commandPool;

    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory vertexBufferMemory = VK_NULL_HANDLE;
    VkBuffer indexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory indexBufferMemory = VK_NULL_HANDLE;

    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;

    VkCommandBuffer commandBuffer;

    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    size_t currentFrame = 0;

    bool framebufferResized = false;

    void initWindow()
    {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
        glfwSetKeyCallback(window, keyCallback);
    }

    void initVulkan()
    {
        createInstance();
        setupDebugCallback();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createGraphicsPipelines();
        createFramebuffers();
        createCommandPool();
        createUniformBuffers();
        createSyncObjects();
    }

    void recreateSwapChain() {
        int width = 0, height = 0;
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(device);

        cleanupSwapChain();

        createSwapChain();
        createImageViews();
        createFramebuffers();
    }

    void mainLoop()
    {
        QueueFamilyIndices familyIndices = findQueueFamilies(physicalDevice);
        VkCommandPool commandPool = createCommandPool(device, familyIndices);

        VkCommandBufferAllocateInfo allocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        allocateInfo.commandPool = commandPool;
        allocateInfo.level =  VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocateInfo.commandBufferCount = 1;

        if (vkAllocateCommandBuffers(device, &allocateInfo, &commandBuffer) != VK_SUCCESS)
            throw std::runtime_error("failed to allocate coomand buffer!");

        VkPhysicalDeviceMemoryProperties memoryProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

        Mesh mesh;
    #if _DEBUG
        bool rcm = loadMesh(mesh, "models/bunny.obj");
    #else
        bool rcm = loadMesh(mesh, "models/happy-buddha.obj");
    #endif
        assert(rcm);

        buildMeshlets(mesh);
        buildMeshletCones(mesh);
        buildMeshletIndices(mesh);

        std::vector<VkDrawIndexedIndirectCommand> indirectCommands;

        // Create on indirect command for each mesh in the scene
		for (size_t i = 0; i < mesh.meshletInstances.size(); i++)
		{
            auto cone = mesh.meshlets[i].cone;
            auto cosangle = glm::dot(glm::vec3(cone[0], cone[1], cone[2]), glm::vec3(0, 0, 1));
            if (cone[3] < cosangle)
                continue;

			VkDrawIndexedIndirectCommand indirectCmd {};
			indirectCmd.instanceCount = 1;
			indirectCmd.firstInstance = uint32_t(i);
			indirectCmd.firstIndex = mesh.meshletInstances[i].first;
			indirectCmd.indexCount = mesh.meshletInstances[i].second;
			
			indirectCommands.push_back(indirectCmd);
		}
        uint32_t indirectDrawCount = uint32_t(indirectCommands.size());

        Buffer scratch = {};
        createBuffer(scratch, device, memoryProperties, 128 * 1024 * 1024, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        Buffer vb = {};
        createBuffer(vb, device, memoryProperties, 128 * 1024 * 1024, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        Buffer ib = {};
        createBuffer(ib, device, memoryProperties, 128 * 1024 * 1024, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        Buffer mb = {};
        createBuffer(mb, device, memoryProperties, 128 * 1024 * 1024, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        Buffer mib = {};
        createBuffer(mib, device, memoryProperties, 128 * 1024 * 1024, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        Buffer cb = {};
        createBuffer(cb, device, memoryProperties, 128 * 1024 * 1024, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        uploadBuffer(device, commandPool, commandBuffer, graphicsQueue, vb, scratch, mesh.vertices.data(), mesh.vertices.size() * sizeof(Vertex));
        uploadBuffer(device, commandPool, commandBuffer, graphicsQueue, ib, scratch, mesh.indices.data(), mesh.indices.size() * sizeof(uint32_t));
        uploadBuffer(device, commandPool, commandBuffer, graphicsQueue, mb, scratch, mesh.meshlets.data(), mesh.meshlets.size() * sizeof(Meshlet));
        uploadBuffer(device, commandPool, commandBuffer, graphicsQueue, mib, scratch, mesh.meshletIndices.data(), mesh.meshletIndices.size() * sizeof(uint32_t));
        uploadBuffer(device, commandPool, commandBuffer, graphicsQueue, cb, scratch, indirectCommands.data(), indirectCommands.size() * sizeof(VkDrawIndexedIndirectCommand));

        VkQueryPool queryPool = createQueryPool(device, 128);

        VkPhysicalDeviceProperties props = {};
        vkGetPhysicalDeviceProperties(physicalDevice, &props);
        assert(props.limits.timestampComputeAndGraphics);

        double frameCpuAvg = 0;
        double frameGpuAvg = 0;
        double frameWaitAvg = 0;

        while (!glfwWindowShouldClose(window)) 
        {
            glfwPollEvents();
            double frameBegin = glfwGetTime()*1000;

            vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, std::numeric_limits<uint64_t>::max());

            uint32_t imageIndex = 0;
            VkResult result = vkAcquireNextImageKHR(device, swapChain, ~0ull, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

            if (result == VK_ERROR_OUT_OF_DATE_KHR) {
                recreateSwapChain();
                continue;
            }
            else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
                throw std::runtime_error("failed_ to acquire swap chain image!");
            }

            if (vkResetCommandPool(device, commandPool, 0) != VK_SUCCESS)
                throw std::runtime_error("failed to reset command pool!");

            VkCommandBufferBeginInfo beginInfo { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

            if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
                throw std::runtime_error("failed to create Begin command buffer!");

            vkCmdResetQueryPool(commandBuffer, queryPool, 0, 128);
            vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 0);

            VkImageMemoryBarrier renderBeginBarrier = imageBarrier(swapChainImages[imageIndex], 0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_DEPENDENCY_BY_REGION_BIT, 0, 0, 0, 0, 1, &renderBeginBarrier);

            VkClearValue clearColor = { glm::sin(float(glfwGetTime()))*0.5f+0.5f, 0.5f, 0.5f, 1.0f };

            VkRenderPassBeginInfo renderPassInfo = {};
            renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            renderPassInfo.renderPass = renderPass;
            renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
            renderPassInfo.renderArea.offset = {0, 0};
            renderPassInfo.renderArea.extent = swapChainExtent;
            renderPassInfo.clearValueCount = 1;
            renderPassInfo.pClearValues = &clearColor;

            vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

            VkViewport viewport = {
                0, float(swapChainExtent.height),
                float(swapChainExtent.width), -float(swapChainExtent.height),
                0.f, 1.f };

            VkRect2D scissor = {};
            scissor.offset = {0, 0};
            scissor.extent = swapChainExtent;

            vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
            vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

            if (bMehsShaderEnabled)
            {
                vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, meshletPipeline);

                DescriptorInfo descriptors[] = { vb.buffer, mb.buffer };
                vkCmdPushDescriptorSetWithTemplateKHR(commandBuffer, meshletUpdateTemplate, meshletPipelineLayout, 0, descriptors);
                vkCmdBindIndexBuffer(commandBuffer, ib.buffer, 0, VK_INDEX_TYPE_UINT32);
                vkCmdDrawMeshTasksNV(commandBuffer, uint32_t(mesh.meshlets.size()), 0);
            }
            else
            {
                vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipeline);
                DescriptorInfo descriptors[] = { vb.buffer };
                vkCmdPushDescriptorSetWithTemplateKHR(commandBuffer, meshUpdateTemplate, meshPipelineLayout, 0, descriptors);

            #if 0
                vkCmdBindIndexBuffer(commandBuffer, ib.buffer, 0, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexed(commandBuffer, uint32_t(mesh.indices.size()), 1, 0, 0, 0);
            #else
                vkCmdBindIndexBuffer(commandBuffer, mib.buffer, 0, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexedIndirect(commandBuffer, cb.buffer, 0, indirectDrawCount, sizeof(VkDrawIndexedIndirectCommand));
            #endif
            }

            vkCmdEndRenderPass(commandBuffer);

            VkImageMemoryBarrier renderEndBarrier = imageBarrier(swapChainImages[imageIndex], VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
            vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_DEPENDENCY_BY_REGION_BIT, 0, 0, 0, 0, 1, &renderEndBarrier);

            vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 1);

            if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
                throw std::runtime_error("failed to end command buffer!");

            VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
            VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };
            VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };

            VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
            submitInfo.waitSemaphoreCount = 1;
            submitInfo.pWaitSemaphores = waitSemaphores;
            submitInfo.pWaitDstStageMask = waitStages;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &commandBuffer;
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = signalSemaphores;

            vkResetFences(device, 1, &inFlightFences[currentFrame]);

            if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
                throw std::runtime_error("failed to submit draw command buffer!");
            }

            VkPresentInfoKHR presentInfo = {};
            presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
            presentInfo.waitSemaphoreCount = 1;
            presentInfo.pWaitSemaphores = signalSemaphores;
            presentInfo.swapchainCount = 1;
            presentInfo.pSwapchains = &swapChain;
            presentInfo.pImageIndices = &imageIndex;
            presentInfo.pResults = nullptr; // optional

            result = vkQueuePresentKHR(presentQueue, &presentInfo);
            if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
                framebufferResized = false;
                recreateSwapChain();
            }
            else if (result != VK_SUCCESS) {
                throw std::runtime_error("failed to present swap chain image!");
            }
            double waitBegin = glfwGetTime() * 1000;;

            // The use of 'VK_QUERY_RESULT_WAIT_BIT' flag in the release mode periodically return an end value that is less than begin
            VK_CHECK(vkDeviceWaitIdle(device));
            uint64_t queryResults[2] = {};
            VK_CHECK(vkGetQueryPoolResults(device, queryPool, 0, ARRAYSIZE(queryResults), sizeof(queryResults), queryResults, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT));

            double waitEnd = glfwGetTime() * 1000;;

            double frameGpuBegin = double(queryResults[0]) * props.limits.timestampPeriod * 1e-6;
            double frameGpuEnd = double(queryResults[1]) * props.limits.timestampPeriod * 1e-6;
            if (frameGpuEnd - frameGpuBegin < 0)
                throw std::runtime_error("failed to ");

            double frameEnd = glfwGetTime() * 1000;;

            frameCpuAvg = glm::mix(frameCpuAvg, (frameEnd - frameBegin), 0.05);
            frameGpuAvg = glm::mix(frameGpuAvg, (frameGpuEnd - frameGpuBegin), 0.05);
            frameWaitAvg = glm::mix(frameWaitAvg, (waitEnd - waitBegin), 0.05);

            char title[256];
            sprintf(title, "cpu: %.2f ms; wait time: %.2f ms; gpu: %.2f ms; triangles %d; meshlets %d", frameCpuAvg, frameWaitAvg, frameGpuAvg, int(mesh.indices.size() / 3), int(mesh.meshlets.size()));
            glfwSetWindowTitle(window, title);
            // currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
        }

        VK_CHECK(vkDeviceWaitIdle(device));
        vkDestroyQueryPool(device, queryPool, nullptr);
        vkDestroyCommandPool(device, commandPool, nullptr);
        
        destroyBuffer(vb, device);
        destroyBuffer(ib, device);
        destroyBuffer(mb, device);
        destroyBuffer(mib, device);
        destroyBuffer(cb, device);
        destroyBuffer(scratch, device);
    }


    void cleanupSwapChain()
    {
        for (auto framebuffer : swapChainFramebuffers)
            vkDestroyFramebuffer(device, framebuffer, nullptr);

        for (auto imageView : swapChainImageViews)
            vkDestroyImageView(device, imageView, nullptr);
        
        vkDestroySwapchainKHR(device, swapChain, nullptr);
    }

    void cleanup() {
        cleanupSwapChain();
        vkDestroyPipeline(device, meshPipeline, nullptr);
        vkDestroyPipeline(device, meshletPipeline, nullptr);
        vkDestroyPipelineLayout(device, meshPipelineLayout, nullptr);
        vkDestroyDescriptorUpdateTemplate(device, meshUpdateTemplate, nullptr);
        vkDestroyDescriptorUpdateTemplate(device, meshletUpdateTemplate, nullptr);
        vkDestroyPipelineLayout(device, meshletPipelineLayout, nullptr);
        vkDestroyRenderPass(device, renderPass, nullptr);
        for (size_t i = 0; i < swapChainImages.size(); i++) {
            vkDestroyBuffer(device, uniformBuffers[i], nullptr);
            vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
        }

        vkDestroyBuffer(device, indexBuffer, nullptr);
        vkFreeMemory(device, indexBufferMemory, nullptr);

        vkDestroyBuffer(device, vertexBuffer, nullptr);
        vkFreeMemory(device, vertexBufferMemory, nullptr);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
            vkDestroyFence(device, inFlightFences[i], nullptr);
        }

        vkDestroyCommandPool(device, commandPool, nullptr);

        vkDestroyDevice(device, nullptr);

        if (enableValidationLayers)
            vkDestroyDebugUtilsMessengerEXT(instance, callback, nullptr);

        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);

        glfwDestroyWindow(window);
        glfwTerminate();
        window = nullptr;
    }

    void createInstance()
    {
        volkInitialize();

        if (enableValidationLayers && !checkValidationLayerSupport())
            throw std::runtime_error("validation layers requested, but not available!");

        VkApplicationInfo appInfo = {};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Hello Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_1;

        VkInstanceCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        auto extensions = getRequiredExtensions();
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }
        else {
            createInfo.enabledLayerCount = 0;
        }

        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS)
            throw std::runtime_error("failed to create instance!");

        volkLoadInstance(instance);

    #ifdef _DEBUG
        uint32_t extensionCount = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> extension(extensionCount);

        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extension.data());
        std::cout << "Available extensions:" << std::endl;

        for (const auto& extension : extension)
            std::cout << "\t" << extension.extensionName << std::endl;
    #endif
    }

    void setupDebugCallback()
    {
        if (!enableValidationLayers) return;

        VkDebugUtilsMessengerCreateInfoEXT createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;

        if (vkCreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &callback) != VK_SUCCESS)
            throw std::runtime_error("failed to set up debug callback!");
    }

    void createSurface()
    {
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
            throw std::runtime_error("failed to create window surface!");
    }
    void pickPhysicalDevice()
    {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        if (deviceCount == 0)
            throw std::runtime_error("failed to find GPUs with Vulkan support!");

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        for (const auto& device : devices) {
            if (isDeviceSuitable(device)) {
                physicalDevice = device;
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE)
            throw std::runtime_error("failed to find a suitable GPU!");
    }

    void createLogicalDevice()
    {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(), indices.presentFamily.value()};

        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo = {};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, availableExtensions.data());
        
        for (auto& extension : availableExtensions)
        {
            if (strcmp(extension.extensionName, "VK_NV_mesh_shader") == 0)
                bMeshShaderSupported = true;
                break;
        }
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

        VkDeviceCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();

        std::vector<const char*> extensions(deviceExtensions);
        if (bMeshShaderSupported)
            extensions.push_back(VK_NV_MESH_SHADER_EXTENSION_NAME);

        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();
        createInfo.pNext = &features;
        features.pNext = &features16;
        features16.pNext = &features8;

        if (bMeshShaderSupported)
            features8.pNext = &featuresMesh;


        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS)
            throw std::runtime_error("failed to create logical device!");

        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
    }

    void createSwapChain() {
        createSwapChain(physicalDevice, surface);
    }

    void createSwapChain(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice, surface);

        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);

        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
            imageCount = swapChainSupport.capabilities.maxImageCount;

        VkSwapchainCreateInfoKHR createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;

        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};

        if (indices.graphicsFamily != indices.presentFamily) {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        }
        else {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;

        createInfo.oldSwapchain = VK_NULL_HANDLE;

        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
            throw std::runtime_error("failed to create swap chain!");
        }

        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
    }

    VkImageView createImageView(VkDevice device, VkImage image)
    {
        VkImageViewCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.image = image;
        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format = swapChainImageFormat;
        createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;

        VkImageView imageView = VK_NULL_HANDLE;
        if (vkCreateImageView(device, &createInfo, nullptr, &imageView) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image views!");
        }
        return imageView;
    }

    void createImageViews() {
        swapChainImageViews.resize(swapChainImages.size());
        for (size_t i = 0; i < swapChainImages.size(); i++) {
            swapChainImageViews[i] = createImageView(device, swapChainImages[i]);
        }
    }

    void createRenderPass() {
        VkAttachmentDescription colorAttachment = {};
        colorAttachment.format = swapChainImageFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference colorAttachmentRef = {};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass = {};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;

        VkRenderPassCreateInfo renderPassInfo = {};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &colorAttachment;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS)
            throw std::runtime_error("failed to create render pass!");
    }

    void createGraphicsPipelines()
    {
        auto fragShaderCode = readFile("shaders/04.meshlet/base.frag.spv");
        auto vertShaderCode = readFile("shaders/04.meshlet/base.vert.spv");
        Shader meshVS = createShader(device, vertShaderCode);
        Shader meshFS = createShader(device, fragShaderCode);

        meshPipelineLayout = createPipelineLayout(device, {&meshVS, &meshFS});
        meshUpdateTemplate = createDescriptorUpdateTemplate(device, VK_NULL_HANDLE, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipelineLayout, {&meshVS, &meshFS});

        VkPipelineCache pipelineCache = VK_NULL_HANDLE;
        meshPipeline = createGraphicsPipeline(device, pipelineCache, renderPass, meshVS, meshFS, meshPipelineLayout);

        if (bMeshShaderSupported)
        {
            auto meshShaderCode = readFile("shaders/04.meshlet/base.mesh.spv");
            Shader meshMS = createShader(device, meshShaderCode);

            meshletPipelineLayout = createPipelineLayout(device, {&meshMS, &meshFS});
            meshletUpdateTemplate = createDescriptorUpdateTemplate(device, VK_NULL_HANDLE, VK_PIPELINE_BIND_POINT_GRAPHICS, meshletPipelineLayout, {&meshMS, &meshFS});

            VkPipelineCache pipelineMeshletCache = VK_NULL_HANDLE;
            meshletPipeline = createGraphicsPipeline(device, pipelineMeshletCache, renderPass, meshMS, meshFS, meshletPipelineLayout);

            destroyShader(device, meshMS);
        }
        destroyShader(device, meshFS);
        destroyShader(device, meshVS);
    }

    VkFramebuffer createFramebuffer(VkDevice device, VkRenderPass renderPass, VkImageView imageView, uint32_t width, uint32_t height)
    {
        VkFramebufferCreateInfo framebufferInfo = {};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = &imageView;
        framebufferInfo.width = width;
        framebufferInfo.height = height;
        framebufferInfo.layers = 1;

        VkFramebuffer framebuffer = VK_NULL_HANDLE;
        if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &framebuffer) != VK_SUCCESS)
            throw std::runtime_error("failed to create framebuffer!");
        return framebuffer;
    }

    void createFramebuffers() {
        swapChainFramebuffers.resize(swapChainImageViews.size());

        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            swapChainFramebuffers[i] = createFramebuffer(device, renderPass, swapChainImageViews[i], swapChainExtent.width, swapChainExtent.height);
        }
    }

    void createCommandPool() {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        VkCommandPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create command pool!");
        }
    }

    VkCommandPool createCommandPool(VkDevice device, QueueFamilyIndices familyIndex)
    {
        VkCommandPoolCreateInfo createInfo = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        createInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        createInfo.queueFamilyIndex = familyIndex.graphicsFamily.value();

        VkCommandPool commandPool = 0;
        if (vkCreateCommandPool(device, &createInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create command pool!");
        }
        return commandPool;
    }

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
        VkBufferCreateInfo bufferInfo = {};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to create buffer!");
        }

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        // TODO: limited to number 4096, so use it with offset etc..
        if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate vertex buffer memory!");
        }

        vkBindBufferMemory(device, buffer, bufferMemory, 0);
    }

    void createBuffer(Buffer& result, VkDevice device, const VkPhysicalDeviceMemoryProperties& memoryPropertices, size_t size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memoryFlags)
    {
        // create buffer object that doesn't have memory back in, just dummy handle
        VkBufferCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        createInfo.size = size;
        createInfo.usage = usage;
        createInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkBuffer buffer;
        if (vkCreateBuffer(device, &createInfo, 0, &buffer) != VK_SUCCESS)
            throw std::runtime_error("failed to createBufer!");

        // how mush memory does use
        VkMemoryRequirements memoryRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memoryRequirements);

        uint32_t memoryTypeIndex = selectMemoryType(memoryPropertices, memoryRequirements.memoryTypeBits, memoryFlags);

        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memoryRequirements.size;
        allocInfo.memoryTypeIndex = memoryTypeIndex;

        VkDeviceMemory memory = 0;
        if (vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS)
            throw std::runtime_error("failed to allocate memory!");

        // ties memory and object together
        if (vkBindBufferMemory(device, buffer, memory, 0) != VK_SUCCESS)
            throw std::runtime_error("failed to bind buffer memory!");

        void* data = nullptr;
        if (memoryFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
            if (vkMapMemory(device, memory, 0, size, 0, &data) != VK_SUCCESS)
                throw std::runtime_error("map memory");

        result.buffer = buffer;
        result.memory = memory;
        result.data = data;
        result.size = size;
    }

    void uploadBuffer(VkDevice device, VkCommandPool commandPool, VkCommandBuffer commandBuffer, VkQueue queue, const Buffer& buffer, const Buffer& scratch, const void* data, size_t size)
    {
        if (scratch.size < size)
            throw std::runtime_error("failed to upload to scratch buffer");
        memcpy(scratch.data, data, size);
        
        VK_CHECK(vkResetCommandPool(device, commandPool, 0));

        VkCommandBufferBeginInfo beginInfo { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));

        VkBufferCopy region = { 0, 0, size };
        vkCmdCopyBuffer(commandBuffer, scratch.buffer, buffer.buffer, 1, &region);

        VkBufferMemoryBarrier copyBarrier = bufferBarrier(buffer.buffer, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_DEPENDENCY_BY_REGION_BIT, 0, 0, 1, &copyBarrier, 0, 0);

        VK_CHECK(vkEndCommandBuffer(commandBuffer));

        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(queue));
    }

    void destroyBuffer(const Buffer& buffer, VkDevice device)
    {
        vkFreeMemory(device, buffer.memory, nullptr);
        vkDestroyBuffer(device, buffer.buffer, nullptr);
    }

    void createUniformBuffers()
    {
        VkDeviceSize bufferSize = sizeof(UniformBufferObject);

        uniformBuffers.resize(swapChainImages.size());
        uniformBuffersMemory.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i]);
        }
    }

    uint32_t selectMemoryType(const VkPhysicalDeviceMemoryProperties& memoryProperties, uint32_t memoryTypeBits, VkMemoryPropertyFlags flags)
    {
        for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++)
            if ((memoryTypeBits & (1 << i)) != 0 && (memoryProperties.memoryTypes[i].propertyFlags & flags) == flags)
                return i;
        throw std::runtime_error("failed to find suitable memory type!");
    }

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) 
    {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

    void createSyncObjects() {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        VkSemaphoreCreateInfo semaphoreInfo = {};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo = {};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
                vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
                vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create synchronization objects for a frame!");
            }
        }
    }

    void updateUniformBuffer(uint32_t currentImage) {
        static auto startTime = std::chrono::high_resolution_clock::now();

        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

        UniformBufferObject ubo = {};
        ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 10.0f);

        void* data = nullptr;
        vkMapMemory(device, uniformBuffersMemory[currentImage], 0, sizeof(ubo), 0, &data);
            memcpy(data, &ubo, sizeof(ubo));
        vkUnmapMemory(device, uniformBuffersMemory[currentImage]);
    }

    void drawFrame()
    {
    }

    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
    {
        if (availableFormats.size() == 1 && availableFormats[0].format == VK_FORMAT_UNDEFINED)
            return {VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};

        for (const auto& availableFormat : availableFormats)
        {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
                return availableFormat;
        }

        return availableFormats.front();
    }

    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes)
    {
        VkPresentModeKHR bestMode = VK_PRESENT_MODE_FIFO_KHR;

        for (const auto& availablePresentMode : availablePresentModes)
        {
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
                return availablePresentMode;
            else if (availablePresentMode == VK_PRESENT_MODE_IMMEDIATE_KHR)
                bestMode = availablePresentMode;
        }

        return bestMode;
    }

    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities)
    {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        }
        else {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);
            VkExtent2D actualExtent = {
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
            };

            actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

            return actualExtent;
        }
    }

    bool isDeviceSuitable(VkPhysicalDevice device) const
    {
        VkPhysicalDeviceProperties deviceProperties = {};
        vkGetPhysicalDeviceProperties(device, &deviceProperties);
        if (deviceProperties.apiVersion < VK_API_VERSION_1_1)
            return false;

        QueueFamilyIndices indices = findQueueFamilies(device);

        bool extensionsSupported = checkDeviceExtensionSupport(device);

        bool swapChainAdequate = false;
        if (extensionsSupported) {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device, surface);
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }

        return indices.isComplete() && extensionsSupported && swapChainAdequate;
    }

    bool checkDeviceExtensionSupport(VkPhysicalDevice device) const
    {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    }

    VkQueryPool createQueryPool(VkDevice device, uint32_t queryCount)
    {
        VkQueryPoolCreateInfo createInfo = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
        createInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        createInfo.queryCount = queryCount;

        VkQueryPool result = VK_NULL_HANDLE;
        VK_CHECK(vkCreateQueryPool(device, &createInfo, 0, &result));
        return result;
    }

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) const
    {
        QueueFamilyIndices indices;

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        int i = 0;
        for (const auto& queueFamily : queueFamilies) {
            if (queueFamily.queueCount > 0 && queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                indices.graphicsFamily = i;
            }

            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

            if (queueFamily.queueCount > 0 && presentSupport)
                indices.presentFamily = i;

            if (indices.isComplete()) {
                break;
            }
            i++;
        }

        return indices;
    }

    std::vector<const char*> getRequiredExtensions()
    {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        if (enableValidationLayers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
        return extensions;
    }

    VkImageMemoryBarrier imageBarrier(VkImage image, VkAccessFlags srcAccessMask, VkAccessFlags dstAccessMask, VkImageLayout oldLayout, VkImageLayout newLayout)
    {
        VkImageMemoryBarrier barrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };

        barrier.srcAccessMask = srcAccessMask;
        barrier.dstAccessMask = dstAccessMask;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        // TODO: android bug can be occur
        barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
        barrier.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;

        return barrier;
    }

    VkBufferMemoryBarrier bufferBarrier(VkBuffer buffer, VkAccessFlags srcAccessMask, VkAccessFlags dstAccessMask)
    {
        VkBufferMemoryBarrier barrier = { VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER };
        barrier.srcAccessMask = srcAccessMask;
        barrier.dstAccessMask = dstAccessMask;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = buffer;
        barrier.offset = 0;
        // TODO: android bug can be occur
        barrier.size = VK_WHOLE_SIZE;

        return barrier;
    }

    bool checkValidationLayerSupport() const
    {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        for (const char* layerName : validationLayers)
        {
            bool layerFound = false;
            for (const auto& layerProperties : availableLayers)
            {
                if (strcmp(layerName, layerProperties.layerName) == 0)
                {
                    layerFound = true;
                    break;
                }
            }
            if (!layerFound)
                return false;
        }
        return true;
    }



    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
        // Validation layers don't correctly track set assignments when using push descriptors with update templates: https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/341
        if (strstr(pCallbackData->pMessage, "uses set #0 but that set is not bound."))
            return VK_FALSE;

        // Validation layers don't correctly detect NonWriteable declarations for storage buffers: https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/73
        if (strstr(pCallbackData->pMessage, "Shader requires vertexPipelineStoresAndAtomics but is not enabled on the device"))
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
            assert(!"Validation error occurs!");
        return VK_FALSE;
    }

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height)
    {
        auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
    {
        auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
        if (action == GLFW_PRESS && key == GLFW_KEY_M)
            app->bMehsShaderEnabled = !app->bMehsShaderEnabled && app->bMeshShaderSupported;
    }
};

int main()
{
    HelloTriangleApplication app;

    try {
        app.run();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}