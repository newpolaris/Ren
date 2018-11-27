#pragma once

#include "common.h"
#include "shaders.h"

VkDescriptorSetLayout createDescriptorSetLayout(VkDevice device, const Shader& vs, const Shader& fs);
VkPipeline createGraphicsPipeline(VkDevice device, VkPipelineCache pipelineCache, VkRenderPass renderPass, const Shader& vs, const Shader& fs, VkPipelineLayout layout);
VkPipelineLayout createPipelineLayout(VkDevice device, const Shader& vs, const Shader& fs);
