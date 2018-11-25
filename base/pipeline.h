#pragma once

#include "common.h"

VkPipelineLayout createPipelineLayout(VkDevice device, bool bMeshShader);
VkPipeline createGraphicsPipeline(VkDevice device, VkPipelineCache pipelineCache, VkRenderPass renderPass, VkShaderModule vs, VkShaderModule fs, VkPipelineLayout layout, bool bMeshShader);
