#include "shaders.h"

VkDescriptorUpdateTemplate createDescriptorUpdateTemplate(VkDevice device, VkDescriptorSetLayout descriptorSetLayout, VkPipelineBindPoint pipelineBindPoint, VkPipelineLayout layout, bool bEnableMeshlet)
{
    std::vector<VkDescriptorUpdateTemplateEntry> entries;

    if (bEnableMeshlet)
    {
        entries.resize(2);
        entries[0].dstBinding = 0;
        entries[0].dstArrayElement = 0;
        entries[0].descriptorCount = 1;
        entries[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        entries[0].offset = 0;
        entries[0].stride = sizeof(VkDescriptorBufferInfo);
        entries[1].dstBinding = 1;
        entries[1].dstArrayElement = 0;
        entries[1].descriptorCount = 1;
        entries[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        entries[1].offset = sizeof(VkDescriptorBufferInfo);
        entries[1].stride = sizeof(VkDescriptorBufferInfo);
    }
    else
    {
        entries.resize(1);
        entries[0].dstBinding = 0;
        entries[0].dstArrayElement;
        entries[0].descriptorCount = 1;
        entries[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        entries[0].offset = 0;
        entries[0].stride = sizeof(VkDescriptorBufferInfo);
    }

    VkDescriptorUpdateTemplateCreateInfo createInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO_KHR};
    createInfo.descriptorUpdateEntryCount = uint32_t(entries.size());
    createInfo.pDescriptorUpdateEntries = entries.data();
    createInfo.templateType = VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_PUSH_DESCRIPTORS_KHR;
    createInfo.descriptorSetLayout = descriptorSetLayout;
    createInfo.pipelineBindPoint = pipelineBindPoint;
    createInfo.pipelineLayout = layout;
    createInfo.set = 0;

    VkDescriptorUpdateTemplate descriptorUpdateTemplate = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDescriptorUpdateTemplate(device, &createInfo, nullptr, &descriptorUpdateTemplate));
    return descriptorUpdateTemplate;
}

VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code) {
    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shader module!");
    }
    return shaderModule;
}
