#pragma once

#include <volk.h>
#include <shader_module.h>

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

struct Program
{
    VkPipelineLayout layout;
    VkPipeline pipeline;
};

VkPipelineLayout CreatePipelineLayout(VkDevice device, ShaderModules shaders);
VkPipeline CreatePipeline(VkDevice device, VkPipelineLayout layout, VkRenderPass pass, ShaderModules shaders);
VkDescriptorUpdateTemplate CreateDescriptorUpdateTemplate(VkDevice device, VkPipelineLayout layout, ShaderModules shaders);
