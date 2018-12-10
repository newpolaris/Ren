#pragma once

#include <volk.h>
#include "spirv-reflect.h"

#include "types.h"

// descriptors that used as data for vkCmdPushDescriptorSetWithTemplateKHR
class PushDescriptorSets
{
public:

    PushDescriptorSets(const Buffer& buffer);
    union {
        VkDescriptorImageInfo image_;
        VkDescriptorBufferInfo buffer_;
    };
};

struct ShaderModule
{   
    VkShaderModule module;
    spirv::SpirvReflections reflections;
};

ShaderModule CreateShaderModule(VkDevice device, const char* filename);

struct Program
{
    VkPipelineLayout layout;
    VkDescriptorUpdateTemplate update;
    VkShaderStageFlags push_constant_stages;
};

Program CreateProgram(VkDevice device, ShaderModules shaders);
void DestroyProgram(VkDevice device, Program* program);

VkPipeline CreatePipeline(VkDevice device, VkPipelineLayout layout, VkRenderPass pass, ShaderModules shaders);
