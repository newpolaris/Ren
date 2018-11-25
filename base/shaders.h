#pragma once

#include "common.h"

VkDescriptorUpdateTemplate createDescriptorUpdateTemplate(VkDevice device, VkDescriptorSetLayout descriptorSetLayout, VkPipelineBindPoint pipelineBindPoint, VkPipelineLayout layout, bool bEnableMeshlet);
VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code);
