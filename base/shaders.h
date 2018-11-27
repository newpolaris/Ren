#pragma once

#include "common.h"

struct Shader
{
    VkShaderModule module_;
    VkShaderStageFlagBits stage_;

    uint32_t storage_buffer_mask_;
};

struct DescriptorInfo
{
    union {
        VkDescriptorBufferInfo buffer_;
        VkDescriptorImageInfo image_;
    };

    DescriptorInfo()
    {
    }

	DescriptorInfo(VkSampler sampler, VkImageView imageView, VkImageLayout imageLayout)
	{
		image_.sampler = sampler;
		image_.imageView = imageView;
		image_.imageLayout = imageLayout;
	}

	DescriptorInfo(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize range)
	{
		buffer_.buffer = buffer;
		buffer_.offset = offset;
		buffer_.range = range;
	}

	DescriptorInfo(VkBuffer buffer)
	{
		buffer_.buffer = buffer;
		buffer_.offset = 0;
		buffer_.range = VK_WHOLE_SIZE;
	}
};

VkDescriptorUpdateTemplate createDescriptorUpdateTemplate(VkDevice device, VkDescriptorSetLayout descriptorSetLayout, VkPipelineBindPoint pipelineBindPoint, VkPipelineLayout layout, const Shader& vs, const Shader& fs);
Shader createShader(VkDevice device, const std::vector<char>& code);
void destroyShader(VkDevice device, Shader& shader);
