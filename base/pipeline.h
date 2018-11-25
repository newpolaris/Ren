#pragma once

#include "common.h"

VkDescriptorSetLayout createDescriptorSetLayout(VkDevice device, bool bMeshShader);
VkPipeline createGraphicsPipeline(VkDevice device, VkPipelineCache pipelineCache, VkRenderPass renderPass, VkShaderModule vs, VkShaderModule fs, VkPipelineLayout layout, bool bMeshShader);
VkPipelineLayout createPipelineLayout(VkDevice device, VkDescriptorSetLayout descriptorSetLayout, bool bMeshShader);
