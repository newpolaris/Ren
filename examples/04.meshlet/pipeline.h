#pragma once

#include "common.h"
#include "shaders.h"

VkDescriptorSetLayout createDescriptorSetLayout(VkDevice device, Shaders shader);
VkPipeline createGraphicsPipeline(VkDevice device, VkPipelineCache pipelineCache, VkRenderPass renderPass, const Shader& vs, const Shader& fs, VkPipelineLayout layout);
VkPipelineLayout createPipelineLayout(VkDevice device, Shaders shader);
