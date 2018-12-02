#include "shaders.h"
#include <vulkan/spirv.h>
#include <istream>
#include <sstream>
#include <string>
#include <map>
#include <unordered_map>

#ifndef POD_GET_MACRO
#define POD_GET_MACRO(Type, Name) \
    Type Name(size_t index) \
    { \
        Type t; \
        get(&t, index); \
        return t; \
    }
#endif

struct StreamReader
{
    StreamReader(const char* data, size_t size) : data_(data), last_(data + size)
    {
    }

    bool eof() const
    {
        return data_ >= last_;
    }

    template <typename T>
    void read(T* t)
    {
        if (data_ + sizeof(T) >= last_)
            throw std::runtime_error("underflow source buffer");

        *t = *reinterpret_cast<const T*>(data_);
        data_ += sizeof(T);
    }

    template <typename T>
    void get(T* t, size_t index) const
    {
        if (data_ + sizeof(T)*index >= last_)
            throw std::runtime_error("underflow source buffer");

        *t = *(reinterpret_cast<const T*>(data_) + index);
    }

    void skip(size_t index)
    {
        data_ += index;
    }

    const char* data_;
    const char* last_;

    POD_GET_MACRO(uint8_t, uint8)
    POD_GET_MACRO(uint16_t, uint16)
    POD_GET_MACRO(uint32_t, uint32)
};

struct Resource
{
};

VkShaderStageFlagBits getShaderStage(SpvExecutionModel executionModel)
{
    switch (executionModel)
    {
        case SpvExecutionModelVertex:
            return VK_SHADER_STAGE_VERTEX_BIT;
        case SpvExecutionModelFragment:
            return VK_SHADER_STAGE_FRAGMENT_BIT;
        case SpvExecutionModelTaskNV:
            return VK_SHADER_STAGE_TASK_BIT_NV;
        case SpvExecutionModelMeshNV:
            return VK_SHADER_STAGE_MESH_BIT_NV;
    }
    throw std::runtime_error("fail to get shader stage");
}

struct VulkanFileHeader { 
    uint32_t VulkanMagicNumber;
    uint32_t VersionNumber;
    uint32_t GeneratorMasicNumber;
    uint32_t Bound;
    uint32_t Reserved;
};

enum SpvTypes : uint32_t
{
    SpvUnknown = 0,
    SpvVariable,
};

namespace intermediate
{
    struct Variable
    {
        uint32_t type_id;
        SpvStorageClass_ storage_class;
    };

    struct Decoration
    {
        SpvDecoration decoration;
        uint32_t operand;
    };
};

struct Variable
{
    uint32_t type_id;
    uint32_t storage_class;
    uint32_t binding;
    uint32_t location;
    uint32_t descriptor_set;
};

void parseShader(Shader& shader, const char* data, size_t size)
{
    StreamReader reader(data, size);

    VulkanFileHeader header = {};
    reader.read(&header);

    assert(header.VulkanMagicNumber == SpvMagicNumber);

    uint32_t idBound = header.Bound;

    std::unordered_map<uint32_t, std::vector<intermediate::Decoration>> decorations;
	std::unordered_map<uint32_t, intermediate::Variable> variables;

    while (!reader.eof())
    {
        uint16_t opcode = reader.uint16(0);
        uint16_t wordCount = reader.uint16(1);

        switch (opcode)
        {
        case SpvOpEntryPoint: 
        {
			assert(wordCount >= 2);
            shader.stage_ = getShaderStage(SpvExecutionModel(reader.uint32(1)));
        } break;
        case SpvOpDecorate:
        {
            assert(wordCount >= 3);
            uint32_t targetid = reader.uint32(1);
            assert(targetid < idBound);

            SpvDecoration decoration = SpvDecoration(reader.uint32(2));
            switch (decoration)
            {
            case SpvDecorationBinding:
            case SpvDecorationDescriptorSet:
            {
                assert(wordCount == 4);
                decorations[targetid].push_back({ decoration, reader.uint32(3) });
            } break;
            }
        } break;
        case SpvOpVariable:
        {
            assert(wordCount >= 4);
            uint32_t resultid = reader.uint32(2);
            assert(resultid < idBound);
            variables.emplace(resultid, intermediate::Variable{ reader.uint32(1), SpvStorageClass_(reader.uint32(3)) });
        } break;
        }

        reader.skip(sizeof(uint32_t)*wordCount);
    }

    std::vector<Variable> result_variables;
    for (auto& var : variables)
    {
        Variable v = {};

        v.type_id = var.second.type_id;
        v.storage_class = var.second.storage_class;

        auto it = decorations.find(var.first);
        if (it == decorations.end())
            continue;

        for (auto& deco : it->second)
        {
            switch (deco.decoration)
            {
            case SpvDecorationBinding:
                v.binding = deco.operand; 
                break;
            case SpvDecorationDescriptorSet:
                v.descriptor_set = deco.operand; 
                break;
            }
        }
        result_variables.push_back(v);
    }

    for (auto& var : result_variables)
    {
        assert(var.descriptor_set == 0);
        assert(var.binding < 32);

        shader.storage_buffer_mask_ |= 1 << var.binding;
    }
}

VkDescriptorUpdateTemplate createDescriptorUpdateTemplate(VkDevice device, VkDescriptorSetLayout descriptorSetLayout, VkPipelineBindPoint pipelineBindPoint, VkPipelineLayout layout, Shaders shaders)
{
    std::vector<VkDescriptorUpdateTemplateEntry> entries;

    uint32_t storageBufferMask = 0;
    for (auto shader : shaders)
        storageBufferMask |= shader->storage_buffer_mask_;
    
    for (uint32_t i = 0; i < 32; ++i)
    {
        if (storageBufferMask & (1 << i))
        {
            VkDescriptorUpdateTemplateEntry entry = {};
            entry.dstBinding = i;
            entry.dstArrayElement = 0;
            entry.descriptorCount = 1;
            entry.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            entry.offset = sizeof(DescriptorInfo) * i;
            entry.stride = sizeof(DescriptorInfo);

            entries.push_back(entry);
        }
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

Shader createShader(VkDevice device, const std::vector<char>& code) {
    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shader module!");
    }
    assert(shaderModule);

    Shader result = {};
    parseShader(result, code.data(), code.size());

    result.module_ = shaderModule;

    return result;
}

void destroyShader(VkDevice device, Shader & shader)
{
    vkDestroyShaderModule(device, shader.module_, nullptr);
}
